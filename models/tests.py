import unittest
import datetime
from data import result_df

class TestData(unittest.TestCase):
    def test_columns(self):
        self.assertEqual(len(result_df.columns), 10)

    def test_null_values(self):
        self.assertFalse(result_df.isnull().sum().sum())

    def test_is_datetime(self):
        x = result_df.index
        print(type(x))
        self.assertTrue(isinstance(x, datetime.date))


if __name__ == '__main__':
    unittest.main()
