import pickle

class GraphParams(dict):
    def __init__(self,
                 N: int,
                 cdegree:int,
                 bdegree: int,
                 ratio: float,
                 prefix: str = "",
                 ideal_auc: float = 0.0,
                 emperical_auc: float = 0.0,
                 clique: bool = False,
                 barabassi: bool = False):
        dict.__init__(self, N=N, 
                      cdegree=cdegree, 
                      bdegree=bdegree, 
                      ratio=ratio,
                      ideal_auc=ideal_auc,
                      emperical_auc=emperical_auc,
                      clique=clique, 
                      barabassi=barabassi)
        self.N = N
        self.cdegree = cdegree
        self.bdegree = bdegree
        self.ratio = ratio
        self.prefix = prefix
        self.barabassi = barabassi
        self.clique = clique
        self.ideal_auc = ideal_auc
        self.emperical_auc = emperical_auc