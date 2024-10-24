import torch
import pytest
from torch.utils.data import DataLoader
from main import CTDataset, MyNeuralNet, train_model, evaluate_model

# Mock data for testing
mock_data_path = 'python_projects/data/test.pt'[0:4]


def test_ctdataset_initialization():
    dataset = CTDataset(mock_data_path)
    assert isinstance(dataset, CTDataset)
    assert len(dataset) > 0


def test_ctdataset_getitem():
    dataset = CTDataset(mock_data_path)
    x, y = dataset[0]
    assert x.shape == (1, 28, 28)
    assert 0 <= x.min() <= x.max() <= 1
    assert y.shape == (10,)
    assert y.sum() == 1


def test_myneuralnet_initialization():
    model = MyNeuralNet()
    assert isinstance(model.Matrix1, torch.nn.Linear)
    assert isinstance(model.Matrix2, torch.nn.Linear)
    assert isinstance(model.Matrix3, torch.nn.Linear)


def test_myneuralnet_forward():
    model = MyNeuralNet()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)


def test_train_model():
    model = MyNeuralNet()
    dataset = CTDataset(mock_data_path)
    dataloader = DataLoader(dataset, batch_size=5)
    trained_model = train_model(dataloader, model, n_epochs=2)
    assert isinstance(trained_model, MyNeuralNet)


def test_evaluate_model():
    model = MyNeuralNet()
    dataset = CTDataset(mock_data_path)
    dataloader = DataLoader(dataset, batch_size=5)
    accuracy = evaluate_model(model, dataloader)
    assert 0 <= accuracy <= 100


def test_main_pipeline():
    # This test would be an integration test of sorts
    # You might need to mock wandb or disable it for testing
    from main import main
    try:
        main()
    except Exception as e:
        pytest.fail(f"Main function raised an exception: {e}")

# Remember to adjust the import statement at the top of the test file to
# correctly import from your main.py file, depending on your project structure.

# Also, note that you'll need to handle the wandb initialization in your tests,
# either by mocking wandb or by adding a condition in your main function to
# skip wandb initialization during testing

# These tests cover the basic functionality of your code. You might want to
# add more specific tests based on your requirements and
# any edge cases you want to handle.
