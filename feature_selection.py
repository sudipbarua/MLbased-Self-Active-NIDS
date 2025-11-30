
import statsmodels.api as sm
import pandas as pd


class FeatureSelection:

    def __init__(self):
        pass

    def backward_elimination(self, df):
        # Backward Elimination
        cols = list(df.columns)
        # Iterate over all columns
        while len(cols) > 0:
            x_1 = df[cols]
            # add a column of constants at beginning of the df
            x_1 = sm.add_constant(x_1)
            model = sm.OLS(df['Label'], x_1).fit()
            p = pd.Series(model.pvalues.values[1:], index=cols)
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if pmax > 0.05:
                cols.remove(feature_with_p_max)
            else:
                break
        selected_features = cols
        print("\nSelected features using Backward elimination method\n")
        print(selected_features)
        return selected_features

    def get_new_feat_df(self, new_feat, df):
        # create new dataframe with the new set of features
        df_nf = pd.DataFrame()
        for col in new_feat:
            df_nf = df_nf.join(other=df[col], how='right')
        return df_nf

    def correl_selection(self, df):
        cor = df.corr()
        # Correlation with output variable
        cor_target = abs(cor['Label'])
        # Selecting highly correlated features
        relevant_features = cor_target[cor_target > 0.05].index
        print("\nSelected features using Pearson's Correlation method\n")
        print(relevant_features)
        return relevant_features
