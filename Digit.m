



[XTrain, YTrain] = digitTrain4DArrayData;
[XTest, YTest]   = digitTest4DArrayData;


layers = [
    imageInputLayer([28 28 1], "Name", "input")

    convolution2dLayer(3, 8, "Padding", "same", "Name", "conv_1")
    batchNormalizationLayer("Name", "bn_1")
    reluLayer("Name", "relu_1")

    maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool_1")

    convolution2dLayer(3, 16, "Padding", "same", "Name", "conv_2")
    batchNormalizationLayer("Name", "bn_2")
    reluLayer("Name", "relu_2")

    maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool_2")

    fullyConnectedLayer(10, "Name", "fc")   % 0–9 için 10 sınıf
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "output")
];


options = trainingOptions("sgdm", ...
    "MaxEpochs", 3, ...               
    "MiniBatchSize", 128, ...
    "InitialLearnRate", 0.01, ...
    "Shuffle", "every-epoch", ...
    "Plots", "training-progress", ...
    "Verbose", false);


net = trainNetwork(XTrain, YTrain, layers, options);


YPred = classify(net, XTest);
acc = mean(YPred == YTest);
fprintf("Test doğruluğu: %.2f %%\n", acc * 100);


idx = randi(size(XTest, 4));       
img = XTest(:, :, 1, idx);
trueLabel = YTest(idx);
predLabel = YPred(idx);

figure;
imshow(img);
title("Gerçek: " + string(trueLabel) + " | Tahmin: " + string(predLabel));
