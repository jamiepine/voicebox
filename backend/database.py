"""
SQLite database ORM using SQLAlchemy.
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import uuid
from pathlib import Path

from . import config

Base = declarative_base()


class VoiceProfile(Base):
    """Voice profile database model."""
    __tablename__ = "profiles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    language = Column(String, default="en")
    avatar_path = Column(String, nullable=True)
    effects_chain = Column(Text, nullable=True)  # JSON-serialized default effects chain
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProfileSample(Base):
    """Voice profile sample database model."""
    __tablename__ = "profile_samples"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    audio_path = Column(String, nullable=False)
    reference_text = Column(Text, nullable=False)


class Generation(Base):
    """Generation history database model."""
    __tablename__ = "generations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    text = Column(Text, nullable=False)
    language = Column(String, default="en")
    audio_path = Column(String, nullable=True)
    duration = Column(Float, nullable=True)
    seed = Column(Integer)
    instruct = Column(Text)
    engine = Column(String, default="qwen")
    model_size = Column(String, nullable=True)
    status = Column(String, default="completed")  # generating, completed, failed
    error = Column(Text, nullable=True)
    is_favorited = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Story(Base):
    """Story database model."""
    __tablename__ = "stories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StoryItem(Base):
    """Story item database model (links generations to stories)."""
    __tablename__ = "story_items"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    story_id = Column(String, ForeignKey("stories.id"), nullable=False)
    generation_id = Column(String, ForeignKey("generations.id"), nullable=False)
    version_id = Column(String, ForeignKey("generation_versions.id"), nullable=True)  # Pin to specific version, null = use generation default
    start_time_ms = Column(Integer, nullable=False, default=0)  # Milliseconds from story start
    track = Column(Integer, nullable=False, default=0)  # Track number (0 = main track)
    trim_start_ms = Column(Integer, nullable=False, default=0)  # Milliseconds trimmed from start
    trim_end_ms = Column(Integer, nullable=False, default=0)  # Milliseconds trimmed from end
    created_at = Column(DateTime, default=datetime.utcnow)


class Project(Base):
    """Audio studio project database model."""
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    data = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GenerationVersion(Base):
    """A version of a generation's audio (clean, processed, alternate takes)."""
    __tablename__ = "generation_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    generation_id = Column(String, ForeignKey("generations.id"), nullable=False)
    label = Column(String, nullable=False)  # "clean", "processed", or user-defined
    audio_path = Column(String, nullable=False)
    effects_chain = Column(Text, nullable=True)  # JSON-serialized effects config, null for clean
    source_version_id = Column(String, ForeignKey("generation_versions.id"), nullable=True)  # Which version was used as input
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EffectPreset(Base):
    """Saved effect chain preset."""
    __tablename__ = "effect_presets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    effects_chain = Column(Text, nullable=False)  # JSON-serialized effects config
    is_builtin = Column(Boolean, default=False)
    sort_order = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)


class AudioChannel(Base):
    """Audio channel (bus) database model."""
    __tablename__ = "audio_channels"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChannelDeviceMapping(Base):
    """Mapping between channels and OS audio devices."""
    __tablename__ = "channel_device_mappings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    channel_id = Column(String, ForeignKey("audio_channels.id"), nullable=False)
    device_id = Column(String, nullable=False)  # OS device identifier


class ProfileChannelMapping(Base):
    """Mapping between voice profiles and audio channels (many-to-many)."""
    __tablename__ = "profile_channel_mappings"
    
    profile_id = Column(String, ForeignKey("profiles.id"), primary_key=True)
    channel_id = Column(String, ForeignKey("audio_channels.id"), primary_key=True)


# Database setup will be initialized in init_db()
engine = None
SessionLocal = None
_db_path = None


def init_db():
    """Initialize database tables."""
    global engine, SessionLocal, _db_path

    _db_path = config.get_db_path()
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{_db_path}",
        connect_args={"check_same_thread": False},
    )

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Run migrations before creating tables
    _run_migrations(engine)
    
    Base.metadata.create_all(bind=engine)
    
    # Create default channel if it doesn't exist
    db = SessionLocal()
    try:
        default_channel = db.query(AudioChannel).filter(AudioChannel.is_default == True).first()
        if not default_channel:
            default_channel = AudioChannel(
                id=str(uuid.uuid4()),
                name="Default",
                is_default=True
            )
            db.add(default_channel)
            
            # Assign all existing profiles to default channel
            profiles = db.query(VoiceProfile).all()
            for profile in profiles:
                mapping = ProfileChannelMapping(
                    profile_id=profile.id,
                    channel_id=default_channel.id
                )
                db.add(mapping)
            
            db.commit()
    finally:
        db.close()

    # Backfill: create "clean" GenerationVersion entries for existing generations
    _backfill_generation_versions()

    # Seed built-in effect presets
    _seed_builtin_presets()


def _run_migrations(engine):
    """Run database migrations."""
    from sqlalchemy import inspect, text
    
    inspector = inspect(engine)
    
    # Check if story_items table exists
    if 'story_items' not in inspector.get_table_names():
        return  # Table doesn't exist yet, will be created fresh
    
    # Get columns in story_items table
    columns = {col['name'] for col in inspector.get_columns('story_items')}
    
    # Migration: Remove position column and ensure start_time_ms exists
    # SQLite doesn't support DROP COLUMN easily, so we recreate the table
    if 'position' in columns:
        print("Migrating story_items: removing position column, using start_time_ms")
        
        with engine.connect() as conn:
            # Check if start_time_ms already exists
            has_start_time = 'start_time_ms' in columns
            
            if not has_start_time:
                # First, add the new column temporarily
                conn.execute(text("ALTER TABLE story_items ADD COLUMN start_time_ms INTEGER DEFAULT 0"))
                
                # Calculate timecodes from position ordering
                result = conn.execute(text("""
                    SELECT si.id, si.story_id, si.position, g.duration
                    FROM story_items si
                    JOIN generations g ON si.generation_id = g.id
                    ORDER BY si.story_id, si.position
                """))
                
                rows = result.fetchall()
                
                current_story_id = None
                current_time_ms = 0
                
                for row in rows:
                    item_id, story_id, position, duration = row
                    
                    if story_id != current_story_id:
                        current_story_id = story_id
                        current_time_ms = 0
                    
                    conn.execute(
                        text("UPDATE story_items SET start_time_ms = :time WHERE id = :id"),
                        {"time": current_time_ms, "id": item_id}
                    )
                    
                    current_time_ms += int(duration * 1000) + 200
                
                conn.commit()
            
            # Now recreate the table without the position column
            # 1. Create new table
            conn.execute(text("""
                CREATE TABLE story_items_new (
                    id VARCHAR PRIMARY KEY,
                    story_id VARCHAR NOT NULL,
                    generation_id VARCHAR NOT NULL,
                    start_time_ms INTEGER NOT NULL DEFAULT 0,
                    created_at DATETIME,
                    FOREIGN KEY (story_id) REFERENCES stories(id),
                    FOREIGN KEY (generation_id) REFERENCES generations(id)
                )
            """))
            
            # 2. Copy data
            conn.execute(text("""
                INSERT INTO story_items_new (id, story_id, generation_id, start_time_ms, created_at)
                SELECT id, story_id, generation_id, start_time_ms, created_at FROM story_items
            """))
            
            # 3. Drop old table
            conn.execute(text("DROP TABLE story_items"))
            
            # 4. Rename new table
            conn.execute(text("ALTER TABLE story_items_new RENAME TO story_items"))
            
            conn.commit()
            print("Migrated story_items table to use start_time_ms (removed position column)")
    
    # Migration: Add track column if it doesn't exist
    # Re-check columns after potential position migration
    columns = {col['name'] for col in inspector.get_columns('story_items')}
    if 'track' not in columns:
        print("Migrating story_items: adding track column")
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE story_items ADD COLUMN track INTEGER NOT NULL DEFAULT 0"))
            conn.commit()
            print("Added track column to story_items")
    
    # Migration: Add trim columns if they don't exist
    # Re-check columns after potential track migration
    columns = {col['name'] for col in inspector.get_columns('story_items')}
    if 'trim_start_ms' not in columns:
        print("Migrating story_items: adding trim_start_ms column")
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE story_items ADD COLUMN trim_start_ms INTEGER NOT NULL DEFAULT 0"))
            conn.commit()
            print("Added trim_start_ms column to story_items")
    
    columns = {col['name'] for col in inspector.get_columns('story_items')}
    if 'trim_end_ms' not in columns:
        print("Migrating story_items: adding trim_end_ms column")
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE story_items ADD COLUMN trim_end_ms INTEGER NOT NULL DEFAULT 0"))
            conn.commit()
            print("Added trim_end_ms column to story_items")

    # Migration: Add avatar_path to profiles table
    if 'profiles' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('profiles')}
        if 'avatar_path' not in columns:
            print("Migrating profiles: adding avatar_path column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE profiles ADD COLUMN avatar_path VARCHAR"))
                conn.commit()
                print("Added avatar_path column to profiles")

    # Migration: Add status and error columns to generations table
    if 'generations' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('generations')}
        if 'status' not in columns:
            print("Migrating generations: adding status column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE generations ADD COLUMN status VARCHAR DEFAULT 'completed'"))
                conn.commit()
                print("Added status column to generations")
        if 'error' not in columns:
            print("Migrating generations: adding error column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE generations ADD COLUMN error TEXT"))
                conn.commit()
                print("Added error column to generations")
        if 'engine' not in columns:
            print("Migrating generations: adding engine column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE generations ADD COLUMN engine VARCHAR DEFAULT 'qwen'"))
                conn.commit()
                print("Added engine column to generations")
        # Re-read columns after engine migration (variable name shadows outer `engine`)
        columns = {col['name'] for col in inspector.get_columns('generations')}
        if 'model_size' not in columns:
            print("Migrating generations: adding model_size column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE generations ADD COLUMN model_size VARCHAR"))
                conn.commit()
                print("Added model_size column to generations")

    # Migration: Add effects_chain to profiles table
    if 'profiles' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('profiles')}
        if 'effects_chain' not in columns:
            print("Migrating profiles: adding effects_chain column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE profiles ADD COLUMN effects_chain TEXT"))
                conn.commit()
                print("Added effects_chain column to profiles")

    # Migration: Add sort_order to effect_presets table
    if 'effect_presets' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('effect_presets')}
        if 'sort_order' not in columns:
            print("Migrating effect_presets: adding sort_order column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE effect_presets ADD COLUMN sort_order INTEGER DEFAULT 100"))
                conn.commit()
                print("Added sort_order column to effect_presets")

    # Migration: Add version_id column to story_items table
    if 'story_items' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('story_items')}
        if 'version_id' not in columns:
            print("Migrating story_items: adding version_id column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE story_items ADD COLUMN version_id VARCHAR"))
                conn.commit()
                print("Added version_id column to story_items")

    # Migration: Add source_version_id to generation_versions table
    if 'generation_versions' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('generation_versions')}
        if 'source_version_id' not in columns:
            print("Migrating generation_versions: adding source_version_id column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE generation_versions ADD COLUMN source_version_id VARCHAR"))
                conn.commit()
                print("Added source_version_id column to generation_versions")

    if 'generations' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('generations')}
        if 'is_favorited' not in columns:
            print("Migrating generations: adding is_favorited column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE generations ADD COLUMN is_favorited BOOLEAN DEFAULT 0"))
                conn.commit()
                print("Added is_favorited column to generations")

    # Migration: Create generation_versions for existing generations
    # (populate after tables are created, handled in init_db)


def _backfill_generation_versions():
    """Create 'clean' version entries for existing generations that don't have any."""
    db = SessionLocal()
    try:
        from pathlib import Path as _Path

        # Find generations that have no version entries
        existing_version_gen_ids = {
            row[0] for row in db.query(GenerationVersion.generation_id).all()
        }
        generations = db.query(Generation).filter(
            Generation.status == "completed",
            Generation.audio_path.isnot(None),
            Generation.audio_path != "",
        ).all()

        count = 0
        for gen in generations:
            if gen.id in existing_version_gen_ids:
                continue
            if not _Path(gen.audio_path).exists():
                continue
            version = GenerationVersion(
                id=str(uuid.uuid4()),
                generation_id=gen.id,
                label="clean",
                audio_path=gen.audio_path,
                effects_chain=None,
                is_default=True,
            )
            db.add(version)
            count += 1

        if count > 0:
            db.commit()
            print(f"Backfilled {count} generation version entries")
    finally:
        db.close()


def _seed_builtin_presets():
    """Ensure built-in effect presets exist in the database."""
    import json
    from .utils.effects import BUILTIN_PRESETS

    db = SessionLocal()
    try:
        for idx, (key, preset_data) in enumerate(BUILTIN_PRESETS.items()):
            sort_order = preset_data.get("sort_order", idx)
            existing = db.query(EffectPreset).filter_by(name=preset_data["name"]).first()
            if not existing:
                preset = EffectPreset(
                    id=str(uuid.uuid4()),
                    name=preset_data["name"],
                    description=preset_data.get("description"),
                    effects_chain=json.dumps(preset_data["effects_chain"]),
                    is_builtin=True,
                    sort_order=sort_order,
                )
                db.add(preset)
            elif existing.sort_order != sort_order:
                existing.sort_order = sort_order
        db.commit()
    finally:
        db.close()


def get_db():
    """Get database session (generator for dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
