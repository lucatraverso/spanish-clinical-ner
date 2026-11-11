from seqeval.metrics import classification_report, f1_score

def evaluate(y_true, y_pred, ignore=("UNCLEAR",)):
    clean = lambda seqs: [["O" if any(t.endswith(i) for i in ignore) else t
                           for t in s] for s in seqs]
    yt, yp = clean(y_true), clean(y_pred)
    print(classification_report(yt, yp, digits=4))
    return f1_score(yt, yp)