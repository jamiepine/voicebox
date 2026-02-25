"""
SQLite database ORM using SQLAlchemy.
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
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
    has_finetune = Column(Boolean, default=False)
    active_adapter_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


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
    audio_path = Column(String, nullable=False)
    duration = Column(Float, nullable=False)
    seed = Column(Integer)
    instruct = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Story(Base):
    """Story database model."""
    __tablename__ = "stories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class StoryItem(Base):
    """Story item database model (links generations to stories)."""
    __tablename__ = "story_items"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    story_id = Column(String, ForeignKey("stories.id"), nullable=False)
    generation_id = Column(String, ForeignKey("generations.id"), nullable=False)
    start_time_ms = Column(Integer, nullable=False, default=0)  # Milliseconds from story start
    track = Column(Integer, nullable=False, default=0)  # Track number (0 = main track)
    trim_start_ms = Column(Integer, nullable=False, default=0)  # Milliseconds trimmed from start
    trim_end_ms = Column(Integer, nullable=False, default=0)  # Milliseconds trimmed from end
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Project(Base):
    """Audio studio project database model."""
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    data = Column(Text)  # JSON string
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AudioChannel(Base):
    """Audio channel (bus) database model."""
    __tablename__ = "audio_channels"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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


class FinetuneSample(Base):
    """Fine-tune training sample database model."""
    __tablename__ = "finetune_samples"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    audio_path = Column(String, nullable=False)
    transcript = Column(Text, nullable=False)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    is_ref_audio = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class FinetuneJob(Base):
    """Fine-tune training job database model."""
    __tablename__ = "finetune_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    status = Column(String, default="pending")  # pending|preparing|training|completed|failed|cancelled
    num_samples = Column(Integer, default=0)
    total_audio_duration_seconds = Column(Float, default=0.0)
    # Training config
    epochs = Column(Integer, default=3)
    learning_rate = Column(Float, default=2e-5)
    batch_size = Column(Integer, default=1)
    lora_rank = Column(Integer, default=32)
    # Progress
    current_epoch = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_loss = Column(Float, nullable=True)
    # Results
    adapter_path = Column(String, nullable=True)
    label = Column(String, nullable=True)  # User-friendly name for the adapter
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


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

    # Migration: Add finetune columns to profiles table
    if 'profiles' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('profiles')}
        if 'has_finetune' not in columns:
            print("Migrating profiles: adding has_finetune column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE profiles ADD COLUMN has_finetune BOOLEAN DEFAULT 0"))
                conn.commit()
                print("Added has_finetune column to profiles")
        if 'active_adapter_path' not in columns:
            print("Migrating profiles: adding active_adapter_path column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE profiles ADD COLUMN active_adapter_path VARCHAR"))
                conn.commit()
                print("Added active_adapter_path column to profiles")

    # Migration: Add label column to finetune_jobs table
    if 'finetune_jobs' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('finetune_jobs')}
        if 'label' not in columns:
            print("Migrating finetune_jobs: adding label column")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE finetune_jobs ADD COLUMN label VARCHAR"))
                conn.commit()
                print("Added label column to finetune_jobs")

    # Migration: Move adapter dirs from adapters/latest/ to adapters/{job_id}/
    _migrate_adapter_dirs(engine)


def _migrate_adapter_dirs(engine):
    """Migrate adapter directories from adapters/latest/ to adapters/{job_id}/.

    Previously all adapters for a profile were saved to a single `latest/`
    directory, overwriting each other.  Now each training job gets its own
    `adapters/{job_id}/` directory so multiple adapters can coexist.
    """
    import shutil
    from sqlalchemy import text

    finetune_dir = config.get_finetune_dir()
    if not finetune_dir.exists():
        return

    with engine.connect() as conn:
        # Find completed jobs whose adapter_path still points to `latest/`
        result = conn.execute(text(
            "SELECT id, profile_id, adapter_path FROM finetune_jobs "
            "WHERE status = 'completed' AND adapter_path IS NOT NULL "
            "AND adapter_path LIKE '%/latest'"
        ))
        rows = result.fetchall()

        for job_id, profile_id, adapter_path in rows:
            old_path = Path(adapter_path)
            if not old_path.exists():
                continue

            new_path = old_path.parent / job_id
            if new_path.exists():
                continue  # Already migrated or collision

            print(f"Migrating adapter dir: {old_path} -> {new_path}")
            shutil.copytree(str(old_path), str(new_path))

            new_path_str = str(new_path)
            conn.execute(text(
                "UPDATE finetune_jobs SET adapter_path = :new_path WHERE id = :job_id"
            ), {"new_path": new_path_str, "job_id": job_id})

            # Update profile's active_adapter_path if it points to the old path
            conn.execute(text(
                "UPDATE profiles SET active_adapter_path = :new_path "
                "WHERE active_adapter_path = :old_path"
            ), {"new_path": new_path_str, "old_path": adapter_path})

        if rows:
            conn.commit()
            # Clean up old `latest/` dirs after all jobs are migrated
            for _, profile_id, adapter_path in rows:
                old_path = Path(adapter_path)
                if old_path.exists() and old_path.name == "latest":
                    shutil.rmtree(str(old_path), ignore_errors=True)
                    print(f"Removed old adapter dir: {old_path}")


def get_db():
    """Get database session (generator for dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
