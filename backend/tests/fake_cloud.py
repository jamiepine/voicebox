"""An in-process fake of the voicebox-cloud API for tests.

Implements the surface the desktop sync client uses — devices, the account-key
escrow, the encrypted object store, the sync feed, and blob storage — behind an
httpx.MockTransport, mirroring apps/api in the voicebox-cloud repo. Every
request body is kept in ``seen_bodies`` so tests can assert what the server was
shown (never key material, never plaintext).
"""

import json

import httpx

STORAGE_HOST = "http://cloud.test/__storage/"


class FakeCloud:
    def __init__(self):
        self.devices: dict[str, dict] = {}
        self.account_key: dict | None = None
        self.objects: dict[str, dict] = {}  # objectId -> row (incl. assets dict)
        self.storage: dict[str, bytes] = {}  # key -> ciphertext
        self.seen_bodies: list[bytes] = []
        self._next_device = 0
        self._next_object = 0
        self._seq = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _ok(data, status=200):
        return httpx.Response(status, json={"ok": True, "data": data})

    def _seq_next(self) -> int:
        self._seq += 1
        return self._seq

    def _find_object(self, kind: str, client_id: str) -> dict | None:
        return next((o for o in self.objects.values() if o["kind"] == kind and o["clientId"] == client_id), None)

    # -- request routing --------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.content:
            self.seen_bodies.append(request.content)
        path, method = request.url.path, request.method

        if path.startswith("/__storage/"):
            key = path[len("/__storage/") :]
            if method == "PUT":
                self.storage[key] = request.content
                return httpx.Response(200)
            data = self.storage.get(key)
            return httpx.Response(200, content=data) if data is not None else httpx.Response(404)

        if path == "/v1/devices" and method == "POST":
            return self._register_device(json.loads(request.content))
        if path == "/v1/devices" and method == "GET":
            return self._ok(list(self.devices.values()))
        if path == "/v1/devices/account-key" and method == "PUT":
            self.account_key = json.loads(request.content)
            return self._ok(None)
        if path == "/v1/devices/account-key" and method == "GET":
            return self._ok(self.account_key)
        if path.startswith("/v1/devices/") and path.endswith("/wrapped-key"):
            device_id = path.split("/")[3]
            if method == "POST":
                self.devices[device_id]["wrappedMasterKey"] = json.loads(request.content)["wrappedMasterKey"]
                return self._ok(None)
            return self._ok({"wrappedMasterKey": self.devices[device_id]["wrappedMasterKey"]})

        if path == "/v1/objects" and method == "POST":
            return self._upsert_object(json.loads(request.content))
        if path.startswith("/v1/objects/") and path.endswith("/commit"):
            return self._commit(path.split("/")[3])
        if path.startswith("/v1/objects/") and method == "DELETE":
            obj = self.objects.get(path.split("/")[3])
            if obj:
                obj["deleted"] = True
                obj["seq"] = self._seq_next()
            return self._ok(None)

        if path == "/v1/sync/changes" and method == "GET":
            return self._changes(request.url.params)

        return httpx.Response(404, json={"ok": False, "error": {"message": f"unhandled {method} {path}"}})

    # -- endpoint implementations ----------------------------------------------

    def _register_device(self, body: dict) -> httpx.Response:
        self._next_device += 1
        device_id = f"dev-{self._next_device}"
        self.devices[device_id] = {
            "id": device_id,
            "name": body["name"],
            "publicKey": body["publicKey"],
            "wrappedMasterKey": None,
            "revokedAt": None,
        }
        return self._ok({"deviceId": device_id, "accountHasKey": self.account_key is not None}, 201)

    def _upsert_object(self, body: dict) -> httpx.Response:
        obj = self._find_object(body["kind"], body["clientId"])
        if obj is None:
            self._next_object += 1
            obj = {
                "id": f"obj-{self._next_object}",
                "kind": body["kind"],
                "clientId": body["clientId"],
                "version": 0,
                "deleted": False,
                "record": None,
                "assets": {},
            }
            self.objects[obj["id"]] = obj

        obj["version"] = max(obj["version"], body["version"])
        obj["seq"] = self._seq_next()
        obj["deleted"] = False

        uploads = []
        record = body.get("record")
        if record and (obj["record"] is None or obj["record"]["hash"] != record["hash"]):
            key = f"o/{obj['id']}/record"
            obj["record"] = {**record, "key": key}
            uploads.append({"for": "record", "key": key, "url": STORAGE_HOST + key})
        for asset in body.get("assets", []):
            caid = asset["clientAssetId"]
            existing = obj["assets"].get(caid)
            if existing is None or existing["hash"] != asset["hash"]:
                key = f"o/{obj['id']}/a/{caid}"
                obj["assets"][caid] = {**asset, "key": key}
                uploads.append({"for": f"asset:{caid}", "key": key, "url": STORAGE_HOST + key})
        return self._ok({"objectId": obj["id"], "seq": obj["seq"], "uploads": uploads}, 201)

    def _commit(self, object_id: str) -> httpx.Response:
        obj = self.objects.get(object_id)
        if obj is None:
            return httpx.Response(404, json={"ok": False, "error": {"message": "object not found"}})
        missing = []
        if obj["record"] and obj["record"]["key"] not in self.storage:
            missing.append("record")
        for caid, asset in obj["assets"].items():
            if asset["key"] not in self.storage:
                missing.append(f"asset:{caid}")
        if missing:
            return httpx.Response(409, json={"ok": False, "error": {"message": f"uploads missing: {missing}"}})
        return self._ok({"objectId": object_id, "committed": True})

    def _changes(self, params) -> httpx.Response:
        since = int(params.get("since", 0))
        limit = int(params.get("limit", 200))
        rows = sorted((o for o in self.objects.values() if o["seq"] > since), key=lambda o: o["seq"])[:limit]
        changes = [
            {
                "id": o["id"],
                "kind": o["kind"],
                "clientId": o["clientId"],
                "version": o["version"],
                "seq": o["seq"],
                "deleted": o["deleted"],
                "record": (
                    {
                        "hash": o["record"]["hash"],
                        "size": o["record"]["size"],
                        "url": STORAGE_HOST + o["record"]["key"],
                    }
                    if o["record"]
                    else None
                ),
                "assets": [
                    {
                        "clientAssetId": caid,
                        "role": a["role"],
                        "hash": a["hash"],
                        "size": a["size"],
                        "url": STORAGE_HOST + a["key"] if a["key"] in self.storage else None,
                    }
                    for caid, a in o["assets"].items()
                ],
            }
            for o in rows
        ]
        cursor = rows[-1]["seq"] if rows else since
        return self._ok({"changes": changes, "cursor": cursor, "hasMore": len(rows) == limit})
