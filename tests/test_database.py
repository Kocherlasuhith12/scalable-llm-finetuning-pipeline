import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.database import Base
from src.database.models import User, Dataset, TrainingJob, ModelRegistry, EvaluationJob, DeploymentEndpoint

TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture
def db_session():
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_user_creation(db_session):
    user = User(username="testuser", email="test@example.com", hashed_password="hashed_pw")
    db_session.add(user)
    db_session.commit()
    
    saved_user = db_session.query(User).filter(User.username == "testuser").first()
    assert saved_user is not None
    assert saved_user.email == "test@example.com"

def test_dataset_and_training_job_relationship(db_session):
    user = User(username="trainer", email="trainer@example.com", hashed_password="hash")
    db_session.add(user)
    db_session.commit()
    
    dataset = Dataset(name="test_data.jsonl", file_path="/tmp/test_data.jsonl", file_type="jsonl", sample_count=100, owner_id=user.id)
    db_session.add(dataset)
    db_session.commit()
    
    job = TrainingJob(
        name="test-lora-job",
        base_model="meta-llama/Llama-3.2-1B",
        dataset_id=dataset.id,
        method="lora",
        hyperparameters={"epochs": 3, "lr": 2e-4},
        owner_id=user.id
    )
    db_session.add(job)
    db_session.commit()
    
    saved_job = db_session.query(TrainingJob).filter(TrainingJob.name == "test-lora-job").first()
    assert saved_job is not None
    assert saved_job.dataset.name == "test_data.jsonl"
