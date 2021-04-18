import unittest
import pandas as pd
import numpy as np
from Column import Column

import logging
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s.%(msecs)03d - %(filename)s:%(lineno)d - %(message)s')


class TestColumn(unittest.TestCase):

    def setUp(self):
        self.ser1 = pd.Series(['abc','def', 'ghij', 'abc'], name='ser1')
        self.ser2 = pd.Series([1, 2, 4],                    name='ser2')
        self.ser3 = pd.Series([1.1, 2.2, 3.3         ],     name='ser3')
        self.ser4 = pd.Series(['true', 'false', 'true', 'false'], name='binary')
    
    def testInitBadParams1(self):
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            f = Column(123)

    def testInitBadParams2(self):            
        # Test bad calls to the constructor.
        with self.assertRaises(TypeError):
            f = Column([])
 
    def testInitGood(self):
        # Check we the constructor returns something sane if passed good params.
        f = Column(self.ser1)

        self.assertIsInstance(f, Column)

  
    def testSize(self):
        f = Column(self.ser1)
        self.assertEqual(f.size, 4)

    def testName(self):
        f = Column(self.ser1)
        self.assertEqual(f.name, 'ser1')        
        f.name = 'changed'
        self.assertEqual(f.name, 'changed')       
        
    def testDtype(self):
        f = Column(self.ser1)
        self.assertEqual(f.dtype, 'object')

    def testNunique(self):
        f = Column(self.ser1)
        self.assertEqual(f.nunique, 3)
        
    def testStrContains(self):
        f = Column(self.ser1)
        
        self.assertTrue(pd.Series.equals(f.StrContains('b'), pd.Series([True, False, False, True])))

    def testStrMatches(self):
        f = Column(self.ser1)
        
        self.assertTrue(pd.Series.equals(f.StrMatches('^a'), pd.Series([True, False, False, True])))
    
    def testIsNumeric(self):
        f1 = Column(self.ser1)
        f2 = Column(self.ser2)
        
        self.assertFalse(f1.IsNumeric())
        self.assertTrue(f2.IsNumeric())
    
    def testMostEtc(self):
        f1 = Column(self.ser1)
        self.assertTrue(f1.Most(f1.StrMatches('^...$'), threshold=0.75))
        self.assertTrue(f1.Some(f1.StrMatches('^...$')))       
        self.assertFalse(f1.All(f1.StrMatches('^...$')))
        self.assertTrue(f1.AFew(f1.StrMatches('^...$')))       
        self.assertFalse(f1.NotAny(f1.StrMatches('^...$')))       
    
    
    def testMakeChild(self):
        f1 = Column(self.ser1)
        f2 = Column(self.ser2)
        
        self.assertEqual(f1.depth, 0)
        self.assertEqual(f2.depth, 0)
        
        self.assertEqual(f1.parent, None)
        self.assertEqual(f2.parent, None)
        
        self.assertEqual(f1.children, [])
        self.assertEqual(f2.children, [])
        
        f1.MakeChild(f2)
        
        self.assertEqual(f1.depth, 0)
        self.assertEqual(f2.depth, 1)
        
        self.assertEqual(f1.parent, None)
        self.assertEqual(f2.parent, f1)
        
        self.assertEqual(f1.children, [f2])
        self.assertEqual(f2.children, [])
        
        self.assertEqual(f1.ChildNames(), ['ser2'])


    def testAncestor(self):
        f1 = Column(self.ser1)
        f2 = Column(self.ser2)   
        f3 = Column(self.ser3)
        
        f1.MakeChild(f2)
        f2.MakeChild(f3)

        self.assertEqual(f1.depth, 0)
        self.assertEqual(f2.depth, 1)
        
        self.assertEqual(f1.parent, None)
        self.assertEqual(f2.parent, f1)
        self.assertEqual(f3.parent, f2)

        self.assertEqual(f1.ancestor, None)
        self.assertEqual(f2.ancestor, f1)
        self.assertEqual(f3.ancestor, f1)   # Note, ancestor not parent.
        
    # Everything is categorical these days!
    def testIsCategorical(self):
        f1 = Column(self.ser1)
        f2 = Column(self.ser2)   
        f3 = Column(self.ser3)     

        self.assertTrue(f1.IsCategorical())
        self.assertTrue(f2.IsCategorical())
        self.assertTrue(f3.IsCategorical())
        
        # Should get an error for an empty dataframe if it has a forced dtype we don't recognise.
        f4 = Column(pd.Series([], dtype='bool'))
        with self.assertRaises(RuntimeError):
            x = f4.IsCategorical()  
       
    def testIsBinary(self):
        f1 = Column(self.ser1)
        f2 = Column(self.ser2)   
        f3 = Column(self.ser3) 
        f4 = Column(self.ser4)

        self.assertFalse(f1.IsBinary())
        self.assertFalse(f2.IsBinary())
        self.assertFalse(f3.IsBinary())
        self.assertTrue(f4.IsBinary())        
        
        # Should get an error for an empty dataframe if it has a forced dtype we don't recognise.
        f4 = Column(pd.Series([], dtype='bool'))
        with self.assertRaises(RuntimeError):
            x = f4.IsBinary()      
    
if __name__ == '__main__':
    unittest.main()
    
