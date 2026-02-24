"""
Tests for profile duplicate name validation.

This test suite verifies that the application correctly handles
duplicate profile names and provides user-friendly error messages.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path to import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Base, VoiceProfile as DBVoiceProfile
from models import VoiceProfileCreate
from profiles import create_profile, update_profile


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    # Create temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"

    # Create engine and session
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()

    yield db

    # Cleanup
    db.close()
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_profiles_dir(monkeypatch, tmp_path):
    """Mock the profiles directory to use a temporary path."""
    import profiles
    monkeypatch.setattr(profiles, '_get_profiles_dir', lambda: tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_create_profile_duplicate_name_raises_error(test_db, mock_profiles_dir):
    """Test that creating a profile with a duplicate name raises a ValueError."""
    # Create first profile
    profile_data_1 = VoiceProfileCreate(
        name="Test Profile",
        description="First profile",
        language="en"
    )

    profile_1 = await create_profile(profile_data_1, test_db)
    assert profile_1.name == "Test Profile"

    # Try to create second profile with same name
    profile_data_2 = VoiceProfileCreate(
        name="Test Profile",
        description="Second profile",
        language="en"
    )

    with pytest.raises(ValueError) as exc_info:
        await create_profile(profile_data_2, test_db)

    # Verify error message is user-friendly
    assert "already exists" in str(exc_info.value)
    assert "Test Profile" in str(exc_info.value)
    assert "choose a different name" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_create_profile_different_names_succeeds(test_db, mock_profiles_dir):
    """Test that creating profiles with different names succeeds."""
    # Create first profile
    profile_data_1 = VoiceProfileCreate(
        name="Profile One",
        description="First profile",
        language="en"
    )

    profile_1 = await create_profile(profile_data_1, test_db)
    assert profile_1.name == "Profile One"

    # Create second profile with different name
    profile_data_2 = VoiceProfileCreate(
        name="Profile Two",
        description="Second profile",
        language="en"
    )

    profile_2 = await create_profile(profile_data_2, test_db)
    assert profile_2.name == "Profile Two"

    # Verify both profiles exist
    assert profile_1.id != profile_2.id


@pytest.mark.asyncio
async def test_update_profile_to_duplicate_name_raises_error(test_db, mock_profiles_dir):
    """Test that updating a profile to a duplicate name raises a ValueError."""
    # Create two profiles with different names
    profile_data_1 = VoiceProfileCreate(
        name="Profile A",
        description="First profile",
        language="en"
    )
    profile_1 = await create_profile(profile_data_1, test_db)

    profile_data_2 = VoiceProfileCreate(
        name="Profile B",
        description="Second profile",
        language="en"
    )
    profile_2 = await create_profile(profile_data_2, test_db)

    # Try to update profile_2 to use profile_1's name
    update_data = VoiceProfileCreate(
        name="Profile A",  # Duplicate name
        description="Updated description",
        language="en"
    )

    with pytest.raises(ValueError) as exc_info:
        await update_profile(profile_2.id, update_data, test_db)

    # Verify error message is user-friendly
    assert "already exists" in str(exc_info.value)
    assert "Profile A" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_profile_keep_same_name_succeeds(test_db, mock_profiles_dir):
    """Test that updating a profile while keeping the same name succeeds."""
    # Create profile
    profile_data = VoiceProfileCreate(
        name="My Profile",
        description="Original description",
        language="en"
    )
    profile = await create_profile(profile_data, test_db)

    # Update profile with same name but different description
    update_data = VoiceProfileCreate(
        name="My Profile",  # Same name
        description="Updated description",
        language="en"
    )

    updated_profile = await update_profile(profile.id, update_data, test_db)

    # Verify update succeeded
    assert updated_profile is not None
    assert updated_profile.id == profile.id
    assert updated_profile.name == "My Profile"
    assert updated_profile.description == "Updated description"


@pytest.mark.asyncio
async def test_update_profile_to_new_unique_name_succeeds(test_db, mock_profiles_dir):
    """Test that updating a profile to a new unique name succeeds."""
    # Create profile
    profile_data = VoiceProfileCreate(
        name="Original Name",
        description="Profile description",
        language="en"
    )
    profile = await create_profile(profile_data, test_db)

    # Update profile with new unique name
    update_data = VoiceProfileCreate(
        name="New Unique Name",
        description="Updated description",
        language="en"
    )

    updated_profile = await update_profile(profile.id, update_data, test_db)

    # Verify update succeeded
    assert updated_profile is not None
    assert updated_profile.id == profile.id
    assert updated_profile.name == "New Unique Name"


@pytest.mark.asyncio
async def test_case_sensitive_names_allowed(test_db, mock_profiles_dir):
    """Test that profile names are case-sensitive (e.g., 'Test' and 'test' are different)."""
    # Create profile with lowercase name
    profile_data_1 = VoiceProfileCreate(
        name="test profile",
        description="Lowercase",
        language="en"
    )
    profile_1 = await create_profile(profile_data_1, test_db)

    # Create profile with different case
    profile_data_2 = VoiceProfileCreate(
        name="Test Profile",
        description="Title case",
        language="en"
    )
    profile_2 = await create_profile(profile_data_2, test_db)

    # Both should succeed since SQLite unique constraint is case-sensitive by default
    assert profile_1.name == "test profile"
    assert profile_2.name == "Test Profile"
    assert profile_1.id != profile_2.id
