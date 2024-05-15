import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    variable_insights_df['Passengers'] = variable_count_groupby['Survived']

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


def plot_total_and_survived_based_on_label(variable_insights_df: pd.DataFrame, variable: str) -> None:
    """_summary_

    Args:
        variable_insights_df (pd.DataFrame): _description_
        variable (str): _description_
    """
    # initializing figure
    plt.figure(tight_layout=True, figsize=(8,6))
    
    # title
    plt.title(f'Numero di Passeggeri Totali e Sopravvissuti in base a {variable}', weight='bold')

    # barplots
    # total
    sns.barplot(data=variable_insights_df, x=variable_insights_df.index, y='Passengers', label='Totale')
    # survived
    sns.barplot(data=variable_insights_df, x=variable_insights_df.index, y='Survived', label='Sopravvissuti')

    # legend
    plt.legend()
    plt.show()


def add_name_title_column(titanic_df: pd.DataFrame, column_with_names: str) -> pd.DataFrame:
    """_summary_

    Args:
        titanic_df (pd.DataFrame): _description_
        column_with_names (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # name example: Braund, Mr. Owen Harris -> we need to extract 'Mr' from the name

    # first split by ',' -> ['Braund', ' Mr. Owen Harris']
    first_split = [x.split(',')[1] for x in titanic_df[column_with_names]]

    # second split by '.' -> [' Mr', ' Owen Harris']
    second_split = [x.split('.')[0] for x in first_split]

    # delete empty space at the beginning using replace -> from ' Mr' to 'Mr'
    title_names = [x.replace(' ', '') for x in second_split]

    # add new column containing titles
    titanic_df['Name_title'] = title_names
    return titanic_df


def find_avg_age_title_names(titanic_df: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        titanic_df (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    # mean groupby 'Name_title'
    _, name_title_mean_groupby, _ = sum_mean_and_count_groupby(train_df=titanic_df, variable='Name_title')

    # extrapolate 'Age' column in the groupy
    name_title_avg_age = name_title_mean_groupby['Age']
    return name_title_avg_age


def tickets_higher_than_1_count(titanic_df: pd.DataFrame, ticket_column: str) -> pd.Series:
    """_summary_

    Args:
        titanic_df (pd.DataFrame): _description_
        ticket_column (str): _description_

    Returns:
        pd.Series: _description_
    """
    # tickets value counts
    ticket_value_counts = titanic_df[ticket_column].value_counts()
    
    # filtering tickets that have value count > 1 (excluding passengers traveling alone)
    tickets_more_than_one = ticket_value_counts[ticket_value_counts > 1]
    return tickets_more_than_one


def delete_null_family_tickets(family_tickets_dict: dict) -> dict:
    """_summary_

    Args:
        family_tickets_dict (dict): _description_

    Returns:
        dict: _description_
    """
    # for loop that iterates through each ticket key in the dictionary
    for ticket in family_tickets_dict:
        # deleting null values from each key
        family_tickets_dict[ticket] = [x for x in family_tickets_dict[ticket] if not pd.isna(x)]
        # excluding empty keys (ones that only had null values)
        family_tickets_dict = {k: v for k, v in family_tickets_dict.items() if v}
    return family_tickets_dict


def family_cabin_list(titanic_df: pd.DataFrame, ticket_column: str) -> dict:
    """_summary_

    Args:
        titanic_df (pd.DataFrame): _description_
        ticket_column (str): _description_

    Returns:
        dict: _description_
    """
    # Series of tickets with more than one value count
    tickets_more_than_one = tickets_higher_than_1_count(titanic_df=titanic_df, ticket_column=ticket_column)
    
    # dictionary to store ticket number and related cabins
    family_tickets_dict = {}
    
    # for loop that iterates through each ticket that has more than one value count
    for ticket in tickets_more_than_one.index:
        # find for each ticket the related cabins
        cabin_series = list(titanic_df.loc[titanic_df[ticket_column] == ticket, 'Cabin'])
        family_tickets_dict[ticket] = cabin_series
        
    # deleting null values
    family_tickets_dict = delete_null_family_tickets(family_tickets_dict=family_tickets_dict)
    return family_tickets_dict


def add_deck_column(titanic_df: pd.DataFrame, cabin_column: str) -> pd.DataFrame:
    """_summary_

    Args:
        titanic_df (pd.DataFrame): _description_
        cabin_column (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # slicing cabin string to retrieve the first letter (which indicates the deck)
    sliced_cabin_name = [list(x) for x in titanic_df[cabin_column]]

    # based on the first element of the cabin name (letter: 'A', 'B', ...) create column indicating the deck
    titanic_df['Cabin_deck'] = [x[0] for x in sliced_cabin_name]
    return titanic_df