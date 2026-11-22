"""Vectorized daily counts, preserving the reference replay's metric schema."""
import numpy as np


class PackedDay:
    def __init__(self, rows, classes, date):
        self.row_identity = id(rows)
        self.date = date
        self.classes = tuple(sorted(classes))
        self.n = len(rows)
        self.scores = np.asarray([r['score'] for r in rows],dtype=np.float64)
        self.labels = np.asarray([r['label'] for r in rows],dtype=np.int64)
        self.prediction = np.asarray([r['predicted'] for r in rows],dtype=np.int64)
        self.known = np.asarray([r['known'] for r in rows],dtype=bool)
        kind = np.asarray([r['kind'] for r in rows],dtype='U10')
        self.disjoint = np.asarray([r.get('source_disjoint',True) for r in rows],dtype=bool)
        if not np.isfinite(self.scores).all() or any(r['date']!=date for r in rows):
            raise ValueError('Invalid cached daily scores or dates')
        if not np.isin(self.prediction,self.classes).all():
            raise ValueError('Unknown predicted class')
        class_index = {c:i for i,c in enumerate(self.classes)}
        self.prediction_index = np.asarray([class_index[int(c)] for c in self.prediction],dtype=np.int64)
        labels, first, self.group_index = np.unique(self.labels,return_index=True,return_inverse=True)
        self.groups = [(bool(self.known[i]),str(kind[i]),int(c)) for c,i in zip(labels,first)]
        for g,(known,category,_) in enumerate(self.groups):
            selected = self.group_index==g
            if not np.all(self.known[selected]==known) or not np.all(kind[selected]==category):
                raise ValueError('One service has inconsistent population coding')
        self.group_order = sorted(range(len(self.groups)),key=lambda i:self.groups[i])
        self.correct = self.known & (self.labels==self.prediction)

    def validate(self, rows, classes, date):
        if self.row_identity!=id(rows) or self.classes!=tuple(classes) or self.date!=date:
            raise ValueError('Packed metrics refer to another score cache')

    def measure(self, cutoffs, isolated=False):
        thresholds = np.asarray([cutoffs[c] for c in self.classes],dtype=np.float64)
        rejected = self.scores > thresholds[self.prediction_index]
        selected = self.disjoint if isolated else np.ones(self.n,dtype=bool)
        groups = self.group_index[selected]
        n = np.bincount(groups,minlength=len(self.groups))
        rejected_n = np.bincount(groups,weights=rejected[selected],minlength=len(self.groups))
        correct_n = np.bincount(groups,weights=(self.correct & ~rejected)[selected],minlength=len(self.groups))
        cells = [dict(known=self.groups[i][0],kind=self.groups[i][1],label=self.groups[i][2],
                      n=int(n[i]),rejected=int(rejected_n[i]),correct_accepted=int(correct_n[i]))
                 for i in self.group_order if n[i]]
        known = [c for c in cells if c['known']]
        heldout = [c for c in cells if c['kind']=='heldout']
        background = [c for c in cells if c['kind']=='background']
        def micro(rows,field):
            count=sum(r['n'] for r in rows)
            return sum(r[field] for r in rows)/count if count else None
        def macro(rows,field):
            return sum(r[field]/r['n'] for r in rows)/len(rows) if rows else None
        return dict(known_n=sum(r['n'] for r in known),known_frr=micro(known,'rejected'),
                    known_macro_frr=macro(known,'rejected'),known_correct_accepted=micro(known,'correct_accepted'),
                    heldout_macro_detection=macro(heldout,'rejected'),background_macro_detection=macro(background,'rejected'),class_counts=cells)
