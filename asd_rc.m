%Data=[split_data.data];
data_label=categorical(label_v3);
person_v3=categorical(person_v3);

Size_buf=length(data_label);
height = 1;
width = 1000;
channels = 19;
sampleSize = 1379;
 
CNN_Data = reshape(Data_v3,[height, width, channels, sampleSize]);
CNN_Labels = categorical(data_label'); 

class = 2;
%person_v3 = data_person;
% X作為training, (100-X)作為testing
indices = crossvalind('HoldOut', person_v3, 8/10) ;
%分出Train_data及Train_data_label
Train_data = CNN_Data(:,:,:,indices == 0);
Train_data_label = CNN_Labels(indices == 0);
%//Train_data = CNN_Data;
%//Train_data_label = CNN_Labels;
%分出Test_data及Test_data_label
Test_data = CNN_Data(:,:,:,indices == 1);
Test_data_label = CNN_Labels(indices == 1);
%//Test_data = test_d;
%//Test_data_label = categorical(test_l');
%把測試資料及標籤以cell形式儲存至test_data {行,列}
%//test_data{1,1} = reshape(Test_data,[height, width, channels, 240]);
test_data{1,1} = Test_data;
test_data{2,1}= Test_data_label';
img = CNN_Data(:,:,:,1); 

layers=[imageInputLayer(size(img))
        convolution2dLayer([1 24],2)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer([1 2],'Stride',2) %stride 2
        
        convolution2dLayer([1 144],2)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer([1 2],'Stride',2) %stride 2

        %convolution2dLayer([1 32],4) 
        %batchNormalizationLayer
        %reluLayer
        %maxPooling2dLayer([1 2],'Stride',2) %stride 2

        %convolution2dLayer([1 12],4) 
        %batchNormalizationLayer
        %reluLayer
        %maxPooling2dLayer([1 2],'Stride',2) %stride 2
        %convolution2dLayer([1 32],72) 
        %batchNormalizationLayer
        %reluLayer
        %maxPooling2dLayer([1 2],'Stride',2) %stride 2
        %convolution2dLayer([1 36],32) 
        %reluLayer
        %maxPooling2dLayer([1 2],'Stride',2) %stride 2

        %convolution2dLayer([1 1],1) 
        %functionLayer(@(X) mean(X))

        %flattenLayer
        

        %lstmLayer(64,'OutputMode','last')

        fullyConnectedLayer(20)
        dropoutLayer(0.35)
        batchNormalizationLayer
         %reluLayer
        %fullyConnectedLayer(25)
        %dropoutLayer(0.4)
        fullyConnectedLayer(class)
        softmaxLayer
        classificationLayer()];

options = trainingOptions('sgdm',... %優化器種類
                          'MaxEpochs',6400,... %訓練完全部資料的次數
                          'InitialLearnRate',0.0002,... %模型學習率
                          'Verbose',false,... %在command window顯示結果
                          'Shuffle','every-epoch', ... %每一個epoch的資料都是隨機開始的
                          'MiniBatchSize',250, ... %每10個資料點就更新一次網路參數
                          'ValidationData',test_data, ... %測試的資料
                          'ValidationFrequency',800 ,... %每個資料點就測試一次
                          'Plots','training-progress'); %畫出網路訓練-測試
                  
lgraph = layerGraph(layers);
figure
plot(lgraph)
net = trainNetwork(Train_data,Train_data_label,layers,options);

[predictedLabels, score] =classify(net, test_data{1,1});

testdata_plot=test_data{2,1};
testdata_plot=testdata_plot';
confusion=plotconfusion(testdata_plot,predictedLabels);