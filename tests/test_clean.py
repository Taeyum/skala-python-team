import pandas as pd
import pytest
from src.clean import clean_nulls

def test_clean_nulls_drop():
    # Create sample dataframe with some nulls
    data = {
        'a': [1.0, 2.0, None, 4.0],
        'b': [5.0, None, 7.0, 8.0],
        'c': ['x', 'y', 'z', 'w']
    }
    df = pd.DataFrame(data)
    
    # Drop strategy
    cleaned = clean_nulls(df, cols=['a', 'b'], strategy='drop')
    
    # Rows with index 0 and 3 should remain
    assert len(cleaned) == 2
    assert float(cleaned.loc[0, 'a']) == 1.0
    assert float(cleaned.loc[3, 'a']) == 4.0

def test_clean_nulls_median():
    # Create sample dataframe with one null
    data = {
        'a': [1.0, 2.0, None, 9.0], # median of 1.0, 2.0, 9.0 is 2.0
        'b': [5.0, 6.0, 7.0, 8.0]
    }
    df = pd.DataFrame(data)
    
    # Median strategy
    cleaned = clean_nulls(df, cols=['a'], strategy='median')
    
    # Null value at index 2 should be filled with median 2.0
    assert len(cleaned) == 4
    assert float(cleaned.loc[2, 'a']) == 2.0
