import pandas as pd


def sum_mean_and_count_groupby(train_df: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        train_df (pd.DataFrame): _description_
        variable (str): _description_

    Returns:
        tuple[[pd.DataFrame, pd.DataFrame, pd.DataFrame]: _description_
    """
    # variable sum groupby
    variable_sum_groupby = train_df.groupby(by=[variable]).sum(numeric_only=True)

    # variable mean groupby
    variable_mean_groupby = train_df.groupby(by=[variable]).mean(numeric_only=True)

    # variable count groupby
    variable_count_groupby = train_df.groupby(by=[variable]).count()
    return variable_sum_groupby, variable_mean_groupby, variable_count_groupby


def get_eda_insights(titanic_train_dataset: pd.DataFrame, variable: str) -> pd.DataFrame:
    """_summary_

    Args:
        titanic_train_dataset (pd.DataFrame): _description_
        variable (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # retrieving sum, mean and count variable groupby
    variable_sum_groupby, variable_mean_groupby, variable_count_groupby = sum_mean_and_count_groupby(train_df=titanic_train_dataset, variable=variable)

    # creating a copy of the sum groupby df to work with
    variable_insights_df = variable_sum_groupby.copy(deep=True)

    # find amount of passengers for each label in the variable
    variable_insights_df['Passengers'] = variable_count_groupby['PassengerId']

    # find percentage of passengers in each label in the variable (with respect to their total amount)
    variable_insights_df['%_Passengers'] = [x / variable_insights_df['Passengers'].sum() for x in variable_insights_df['Passengers']]
    
    # find percentage of survived passengers for each label in the variable
    variable_insights_df['%_Sopravvivenza'] = variable_mean_groupby['Survived']

    # find average age of each label in the variable
    variable_insights_df['Età_media'] = variable_mean_groupby['Age']

    # find average ticket fare of each label in the variable
    variable_insights_df['Prezzo_medio'] = variable_mean_groupby['Fare']

    # re-ordering columns
    variable_insights_df = variable_insights_df[['Passengers', '%_Passengers', 'Survived', '%_Sopravvivenza', 'Età_media', 'Prezzo_medio']]
    return variable_insights_df
