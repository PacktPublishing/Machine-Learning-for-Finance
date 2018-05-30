import marbles.core
from marbles.mixins import mixins

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AgeTestCase(marbles.core.TestCase,mixins.DateTimeMixins):
    def setUp(self):
        self.df = pd.DataFrame({'parent':[datetime(1959,7,12)],
                                'child':[datetime(1800,1,1)]},
                                index=[0])
        
    def tearDown(self):
        self.df = None
        
    def test_parent_older_child(self):
        self.assertDateTimesBefore(sequence=self.df.parent,
                                    target=self.df.child,
                                    note='Parents have to be born after their children')
        
    def test_old_age(self):
        max_td = timedelta(365*100)
        today = datetime.today()
        
        self.assertDateTimesAfter(sequence=self.df.child,target=today-max_td)
        
if __name__ == '__main__':       
    marbles.core.main()