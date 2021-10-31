from mlxtend.frequent_patterns import apriori, association_rules
from Helper.eda import *
from Helper.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


def create_invoice_product_df(dataframe, ID=False):
    if ID:
        return dataframe.pivot_table(values="Quantity", columns="StockCode", index="Invoice", aggfunc="sum"). \
            fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.pivot_table(values="Quantity", columns="Description", index="Invoice", aggfunc="sum"). \
            fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def create_rules(dataframe, ID=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, ID)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    return association_rules(frequent_itemsets, metric="support", min_threshold=0.01)


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].toString()
    print(product_name)


# Data Preparation
retail = load_data("online_retail_II.xlsx")
df = retail.copy()
check_df(df)
df = retail_data_prep(df)
check_df(df)

# Creating association rules based on Germany customers.
rules = create_rules(df, country="Germany")
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head()

#  Names of products whose IDs are given
product_id_list = [21987, 23235, 22747]
for product_id in product_id_list:
    print(product_id)
    check_id(df, stock_code=product_id)


# Product recommendation for users in the cart
def arl_recommender(rules_df, productID, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == productID:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


recommendation_lists = []
for product_id in product_id_list:
    recommendation_lists.append(arl_recommender(rules, product_id, rec_count=5))

# Recommended products
counter = 1
for recommendation in recommendation_lists:
    print(f"recommendations list {counter}")
    for item in recommendation:
        check_id(df, item)
    counter += 1
