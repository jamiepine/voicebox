"""Folder tree helpers shared by the folders routes and history queries.

Clip folders nest, so both "list this folder's clips" and "don't let a
folder be moved inside itself" need the same subtree walk.  It lives here
so the two callers can't drift apart.
"""

from sqlalchemy.orm import Session

from ..database.models import Folder


def folder_and_descendants(folder_id: str, db: Session) -> set[str]:
    """Return ``folder_id`` plus every folder beneath it.

    Breadth-first over ``parent_id``.  The seen-set both prevents revisiting
    shared subtrees and stops a cycle -- which the reparent guard should make
    impossible, but which a hand-edited database could still contain -- from
    looping forever.
    """
    seen = {folder_id}
    frontier = [folder_id]

    while frontier:
        rows = db.query(Folder.id).filter(Folder.parent_id.in_(frontier)).all()
        frontier = [child_id for (child_id,) in rows if child_id not in seen]
        seen.update(frontier)

    return seen
