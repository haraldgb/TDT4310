#%%
import pandas as pd

# %%
data = "amazon_reviews_us_Major_Appliances_v1_00.tsv"
df = pd.read_csv(data, sep='\t', error_bad_lines=False)
df.head()
# %%
columns_to_keep = ['star_rating', 'helpful_votes', 'total_votes', 'verified_purchase', 'review_headline', 'review_body']
df = df[columns_to_keep]
df.head()
df.to_csv('amazon_appliances_reviews.csv', index=False)
# %%
df.describe()
df.nunique()

import matplotlib.pyplot as plt
stars = df['star_rating'].value_counts()
plt.bar(stars.index, stars.values)
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.show()