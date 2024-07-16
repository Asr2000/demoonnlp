% Sample text data
textData = ["I love the new features of MATLAB.", ...
            "The interface is user-friendly.", ...
            "Sometimes the processing speed is slow.", ...
            "Overall, it is a great tool for data analysis."];

% Convert to lowercase
textData = lower(textData);

% Remove punctuation
textData = erasePunctuation(textData);

% Tokenize the text
documents = tokenizedDocument(textData);

% Display the tokenized documents
disp(documents);
% Remove stop words
documents = removeStopWords(documents);

% Display the cleaned documents
disp(documents);
% Create a word cloud
figure;
wordcloud(documents);
title('Word Cloud of Sample Text Data');
% Sample text data and corresponding labels (1 for positive, -1 for negative)
trainingData = ["I love MATLAB.", "The interface is great.", "The software is slow.", "It is a fantastic tool."];
labels = [1, 1, -1, 1];

% Convert to lowercase and remove punctuation
trainingData = lower(trainingData);
trainingData = erasePunctuation(trainingData);

% Tokenize the text
trainDocuments = tokenizedDocument(trainingData);

% Remove stop words
trainDocuments = removeStopWords(trainDocuments);

% Create a bag-of-words model
bag = bagOfWords(trainDocuments);

% Train a Naive Bayes classifier
mdl = fitcnb(bag.Counts, labels);

% Predict sentiments for new data
newData = ["I love the new interface.", "The tool is too slow."];
newData = lower(newData);
newData = erasePunctuation(newData);
newDocuments = tokenizedDocument(newData);
newDocuments = removeStopWords(newDocuments);
newBag = bagOfWords(newDocuments);

% Predict sentiments
predictedSentiments = predict(mdl, newBag.Counts);

% Display the predicted sentiments
disp(predictedSentiments);
% Example text for NER
text = "MATLAB is developed by MathWorks. It is used by millions of engineers and scientists.";

% Tokenize the text
document = tokenizedDocument(text);

% Perform Named Entity Recognition
entities = entityRecognition(document);

% Display the recognized entities
disp(entities);
% Sample text data and corresponding sentiment scores
textData = ["I love MATLAB.", "The interface is great.", "The software is slow.", "It is a fantastic tool."];
sentiments = [1, 1, -1, 1];

% Bar chart of sentiment scores
figure;
bar(categorical(textData), sentiments);
title('Sentiment Analysis Results');
ylabel('Sentiment Score');
xlabel('Text Data');
