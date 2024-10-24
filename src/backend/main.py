import matplotlib.pyplot as plt
import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .main import train_accuracy, test_accuracy

app = FastAPI()

def generate_plot(train_acc, test_acc):
    # Example data: replace this with your model's output
    epochs = range(1, 21)  # Assuming you have 20 epochs
    train_loss = train_acc
    test_accuracy = test_acc

    plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Training Loss and Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer to the beginning
    plt.close()  # Close the plot to free memory
    return buf

@app.get("/plot")
async def get_plot():
    buf = generate_plot(train_accuracy, test_accuracy)
    return StreamingResponse(buf, media_type="image/png")
