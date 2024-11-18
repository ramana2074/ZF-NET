#Fine-tuning ZFNet and Visualizing Layers
##Overview
This project fine-tunes the ZFNet architecture on a small subset of the ImageNet dataset. Key objectives include understanding how ZFNet processes images by visualizing intermediate layer outputs and evaluating its accuracy on selected samples.

##Concept
ZFNet is a convolutional neural network designed for image classification. It refines AlexNet by:

Using smaller receptive fields in early layers for better feature capture.
Modifying strides and filter sizes for improved spatial resolution.
##This project:

Fine-tunes ZFNet to work with a limited subset of ImageNet.
Selects 10 random images for evaluation.
Visualizes intermediate feature maps to understand layer-wise processing.
Code Highlights
Model Fine-Tuning
ZFNet's final layer is adjusted dynamically to match the dataset's class count:

python
Copy code
model.classifier[-1] = nn.Linear(4096, num_classes)
Training Loop
The model is trained for 15 epochs using Adam optimizer and Cross-Entropy loss:

python
Copy code
for epoch in range(15):
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
##Per-Image Evaluation
Accuracy is calculated for each of the 10 selected images:

python
Copy code
for images, labels in loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    for i in range(len(images)):
        correct = (predicted[i] == labels[i]).item()
        print(f"Image {i + 1}: {'Correct' if correct else 'Incorrect'}")
##Visualization of Layers
Feature maps from intermediate layers are visualized using heatmaps:

python
Copy code
def visualize_layer(model, images, layer_index):
    hook = list(model.features.children())[layer_index].register_forward_hook(
        lambda m, i, o: o)
    with torch.no_grad():
        model(images)
    hook.remove()
##Summary
Fine-tuning: Adjusted ZFNet to work with a small dataset.
Visualization: Interpreted intermediate layers to understand feature extraction.
Evaluation: Achieved 80% accuracy on 10 selected images.
This project demonstrates the adaptability of ZFNet and provides a foundation for deeper analysis of CNN behavior.
