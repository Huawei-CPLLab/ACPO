import sys
import unittest

import numpy as np
import pandas as pd
from numpy.testing import *
from pandas.testing import *

sys.path.append("../src")
from csv_process import *

# set data for testing purposes ----------------------------------
data = pd.DataFrame(
    columns=list("AB"),
    data=np.array([[1, 1, 1, 2, 2, 1, 2, 1], [1, 2, 3, 1, 2, 2, 3, 2]]).T,
)
data["C"] = pd.Series([0, 2, 2, 4, 2, 8, 16, 4])
data["D"] = pd.Series([True, False, np.nan, True, True, False, False, False])
data["E"] = pd.Series(["Yes", "No", "Maybe", "Maybe", "Yes", "Yes", np.nan, "Yes"])
data["F"] = 13
data["G"] = "OK"
data["R"] = pd.Series([4.2, 6.9, np.nan, 2.4, 7.5, 3.8, 2.2, 8.4])
data["S"] = pd.Series([1.2, 0.9, 0.8, 1.4, 1.5, 1.8, 1.2, 1.4])
data.loc[8] = data.loc[4]

rows = set(range(9))

# ----------------------------------------------------------------


class TestCSVProcessMethod(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_remove_duplicates(self):

        df = data.copy()
        remove_duplicates(df)
        assert_frame_equal(df, data.loc[rows - {4}])

        df = data.copy()
        remove_duplicates(df, keys="all")
        assert_frame_equal(df, data.loc[rows - {4}])

        df = data.copy()
        remove_duplicates(df, keys=list("AB"))
        assert_frame_equal(df, data.loc[rows - {4, 1, 5}])

        df = data.copy()
        remove_duplicates(df, keys=list("AB"), order_key="S")
        assert_frame_equal(df, data.loc[rows - {4, 1, 7}])

        df = data.copy()
        remove_duplicates(df, keys=list("AB"), order_key="S", keep="last")
        assert_frame_equal(df, data.loc[rows - {4, 1, 7}])

        df = data.copy()
        remove_duplicates(df, keys=list("AB"), order_key="S", keep="first")
        assert_frame_equal(df, data.loc[rows - {5, 7, 8}])

        df = data.copy()
        remove_duplicates(df, keys=list("AB"), order_key="S", keep="max")
        assert_frame_equal(df, data.loc[rows - {4, 1, 7}])

        df = data.copy()
        remove_duplicates(
            df, keys=list("AB"), order_key="S", ascending=False, keep="last"
        )
        assert_frame_equal(df, data.loc[rows - {4, 5, 7}])

        df = data.copy()
        remove_duplicates(
            df, keys=list("AB"), order_key="S", ascending=False, keep="first"
        )
        assert_frame_equal(df, data.loc[rows - {1, 7, 8}])

        df = data.copy()
        remove_duplicates(
            df, keys=list("AB"), order_key="S", ascending=False, keep="max"
        )
        assert_frame_equal(df, data.loc[rows - {4, 5, 7}])

    def test_get_index(self):

        df = data.copy()

        idx = get_index(df, "S", "<", 1.4)
        self.assertEqual(idx.values.tolist(), [0, 1, 2, 6])

        idx = get_index(df, "S", ">", 1.5)
        self.assertEqual(idx.values.tolist(), [5])

        idx = get_index(df, "S", "=", 1.4)
        self.assertEqual(idx.values.tolist(), [3, 7])

        idx = get_index(df, "S", "==", 1.4)
        self.assertEqual(idx.values.tolist(), [3, 7])

        idx = get_index(df, "F", "!", 13)
        self.assertEqual(idx.values.tolist(), [])

        idx = get_index(df, "C", "~", 2)
        self.assertEqual(idx.values.tolist(), [0, 3, 5, 6, 7])

        idx = get_index(df, "S", ">=", 1.5)
        self.assertEqual(idx.values.tolist(), [4, 5, 8])

        idx = get_index(df, "R", "<=", 3)
        self.assertEqual(idx.values.tolist(), [3, 6])

    def test_prune_data(self):

        df = data.copy()
        prune_data(df, key="E", values=["Maybe", "No"])
        assert_frame_equal(df, data.loc[rows - {1, 2, 3}])

        df = data.copy()
        prune_data(df, key="E", values="Maybe", queries=["S < 1."])
        assert_frame_equal(df, data.loc[rows - {2}])

        df = data.copy()
        prune_data(df, queries=["S < 1.4", "D == True"])
        assert_frame_equal(df, data.loc[rows - {0}])

        df = data.copy()
        prune_data(df, key="E", values=["Maybe", "No"], queries=["S < 1."])
        assert_frame_equal(df, data.loc[rows - {1, 2}])

        df = data.copy()
        prune_data(df, key="E", values=["Maybe", "No"], queries=["A == 1.", "C == 4"])
        assert_frame_equal(df, data.loc[rows])

        df = data.copy()
        prune_data(
            df,
            key="E",
            values=["Maybe", "No"],
            queries=["A == 1.", "C == 4"],
            logic="OR",
        )
        assert_frame_equal(df, data.loc[rows - {1, 2, 3}])

    def test_drop_feature(self):

        df = data.copy()
        drop_feature(df, keys="F")
        assert_index_equal(df.columns, data.columns.drop("F"))

        df = data.copy()
        drop_feature(df, keys="K")
        assert_index_equal(df.columns, data.columns)

        df = data.copy()
        drop_feature(df, keys=list("ESK"))
        assert_index_equal(df.columns, data.columns.drop(list("ES")))

    def test_add_feature(self):

        df, df2 = data.copy(), data.copy()
        add_feature(df, key="K", value=list(range(30, 39)))
        df2["K"] = list(range(30, 39))
        assert_frame_equal(df, df2)

        df, df2 = data.copy(), data.copy()
        add_feature(df, key="K", value=73)
        df2["K"] = 73
        assert_frame_equal(df, df2)

        df, df2 = data.copy(), data.copy()
        loc = len(df2.columns) - 1
        add_feature(df, key="K", value=25, loc=loc)
        df2.insert(loc=loc, column="K", value=25)
        assert_frame_equal(df, df2)

    def test_find_constant_feature(self):

        df = data.copy()
        cols = find_constant_feature(df)
        self.assertEqual(cols, ["F"])

        cols = find_constant_feature(df, alphanum=False)
        self.assertEqual(cols, ["F"])

        cols = find_constant_feature(df, alphanum=True)
        self.assertEqual(cols, ["F", "G"])

    def test_remove_constant_feature(self):

        df = data.copy()
        remove_constant_feature(df, debug=True)
        assert_index_equal(df.columns, data.columns.drop("F"))

    def test_find_null(self):

        df = data.copy()
        idx = find_null(df)
        self.assertEqual(idx, [2, 6])

        remove_null(df)
        idx = find_null(df)
        self.assertEqual(idx, [])

    def test_remove_null(self):

        df = data.copy()
        remove_null(df, debug=True)
        assert_frame_equal(df, data.loc[rows - {2, 6}])

    def head_count(self):

        df = data.copy()
        self.assertEqual(head_count(df, subset="D")["count"].unique(), 4)
        self.assertEqual(head_count(df, subset="D")["count"], [5, 2, 2])

    def test_feature_to_class(self):

        df = data.copy()
        df, cats = feature_to_class(df, keys=list("AB"))
        keys = [(i, j) for i in [1, 2] for j in [1, 2, 3]]
        key_to_class = {key: i for i, key in enumerate(keys)}
        self.assertEqual(cats, keys)
        assert_array_equal(
            df["Classes"].values,
            [key_to_class[(i, j)] for (i, j) in zip(data["A"], data["B"])],
        )

        df = data.copy()
        df1, cat1 = feature_to_class(df, keys=list("AB"))
        df2, cat2 = feature_to_class(df, keys=list("AB"), cat=cat1)
        self.assertEqual(cat1, cat2)
        assert_frame_equal(df1, df2)

        df = data.copy()
        df1, cat1 = feature_to_class(df, keys=list("AB"))
        df2, cat2 = feature_to_class(df, keys=list("AB"))
        self.assertEqual(cat1, cat2)
        assert_frame_equal(df1, df2)

    def test_feature_relation(self):

        df = data.copy()
        self.assertEqual(features_relation(df).shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
