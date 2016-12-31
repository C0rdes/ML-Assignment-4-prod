
testdata = 'datasets\iristestml.dt';
traindata = 'datasets\iristrainml.dt';

TestM = dlmread(testdata);
TrainM = dlmread(traindata);

[E, w, b] = LogisticRegression(TrainM, 0.2);

zeroOneLossTrain = logitZeroOneLoss(w, b, TrainM);
zeroOneLossTest = logitZeroOneLoss(w, b, TestM);