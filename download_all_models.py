import morpheus


model_map = morpheus.GPTNeo(model=None)
print(model_map.model_map)
for model_selected in model_map.model_map:
    print("Downloading model: " + model_selected)
    model_class = morpheus.GPTNeo(model=model_selected)