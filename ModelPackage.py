class ModelPackage(object):
    def __init__(self, wordEmbeddings, caseEmbeddings, word2Idx, label2Idx, char2Idx, modelName, dataset, epochs=50):
        super(ModelPackage,  self).__init__()
        self.epochs = epochs
        self.wordEmbeddings = wordEmbeddings
        self.caseEmbeddings = caseEmbeddings
        self.word2Idx = word2Idx
        self.label2Idx = label2Idx
        self.char2Idx = char2Idx
        self.modelName = modelName
        self.dataset = dataset
        self.model = None
        
    
    def setModel(self, model):
        self.model = model
        
