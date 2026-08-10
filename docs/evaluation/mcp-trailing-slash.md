# Compatibilidad MCP con `/mcp` y `/mcp/`

## 1. Origen del problema

Clientes MCP que configuraban la URL sin barra final recibían `405 Method Not
Allowed` al enviar `POST /mcp`; el mismo handshake funcionaba en
`POST /mcp/`.

## 2. Causa raíz técnica

FastMCP registra su endpoint interno en `/`. Al montarlo bajo `/mcp`,
Starlette entrega la ruta vacía al sub-application cuando se usa el mount root
sin barra. Esa ruta no coincide con `/` y el POST no llega al handler MCP.

## 3. Fix aplicado

`MountRootSlashRewrite` normaliza una ruta interna vacía a `/` antes de delegar
en FastMCP. Se usa tanto en `backend/app.py` como en `mount_into()`. No se usa
un redirect 307 porque algunos clientes MCP no repiten POST tras redirects.

La documentación y `.mcp.json` usan `/mcp/`, mientras el servidor acepta ambas
formas.

## 4. Verificación

```bash
python -m pytest backend/tests/test_mcp_mount_slashes.py -q
```

El test monta solo FastMCP, activa su lifespan mediante `TestClient` como
context manager, y envía el handshake `initialize` a `/mcp` y `/mcp/`.

## 5. Estado upstream

Pendiente de ejecutar con las dependencias FastAPI/FastMCP del proyecto y de
probar contra el cliente MCP real. El test no necesita GPU ni modelos.
