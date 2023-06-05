def load_model(model_name, num_classes=10):
    if model_name == 'simcnn':
        from models.simcnn import SimCNN
        model = SimCNN(num_classes=num_classes)
    elif model_name == 'rescnn':
        from models.rescnn import ResCNN
        model = ResCNN(num_classes=num_classes)
    else:
        raise ValueError
    return model
