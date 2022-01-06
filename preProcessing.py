from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preProcess(data):
    #drop data that has no grade
    idx = 0
    nf = []
    for i in data['ProductGrade']:
        if not (i == 'A' or i == 'B' or i == 'C' or i == 'D'):
            nf.append(idx)
        idx += 1
    data = data.drop(nf, axis=0)

    data = data.dropna(subset=['uniq_id', 'ProductGrade', 'product_name', 'price'])
    data['number_available_in_stock'] = data['number_available_in_stock'].str.replace('new', '')
    data['price'] = pd.to_numeric(data['price'], downcast='float', errors='coerce')
    data['number_available_in_stock'] = pd.to_numeric(data['number_available_in_stock'], downcast='float', errors='coerce')
    data['number_of_reviews'] = pd.to_numeric(data['number_of_reviews'], downcast='float', errors='coerce')
    data['number_of_answered_questions'] = pd.to_numeric(data['number_of_answered_questions'], downcast='float', errors='coerce')
    #print(data['price'])
    for i in data:
        random_sample = data[i].dropna().sample(data[i].isnull().sum())
        random_sample.index = data[data[i].isnull()].index
        data.loc[data[i].isnull(), i] = random_sample
    data = data.drop('uniq_id', axis=1)
    data = data.drop('product_name', axis=1)
    data = data.drop('product_information', axis=1)

    #classify categotries into two features: first and second categories
    categories = []
    for i in data['amazon_category_and_sub_category']:
        cur = i.split('>')
        categories.append(cur)
    tmp = pd.DataFrame(categories, columns=['category1', 'category2', 'category3', 'category4', 'category5'])
    data['category1'] = tmp['category1']
    data['category2'] = tmp['category2']
    data = data.drop('amazon_category_and_sub_category', axis=1)

    # classify sellers into: seller names and sellers prices
    sellers = []
    cnt = 0
    for i in data['sellers']:
        cnt += 1
        if cnt % 100 == 0:
            print('Pre-Processing', int((cnt / len(data['sellers'])) * 100), '%')
        cur = i.split('{')
        if len(cur) <= 2:
            continue
        arr = []
        for j in range(2, min(5, len(cur))):
            cur2 = cur[j].split("=>")
            #print(cur2)
            name_price = cur2[1].split(',')
            arr.append(name_price[0])
            arr.append(cur2[2])
        sellers.append(arr)
        for j in sellers:
            for l in range(1, len(j), 2):
                price = ''
                for k in j[l]:
                    if(k >= '0' and k <= '9' or (k == '.')):
                        price += k
                j[l] = price
    tmp = pd.DataFrame(sellers, columns=['seller_name_1', 'seller_price_1', 'seller_name_2', 'seller_price_2', 'seller_name_3', 'seller_price_3'])
    data['seller_name_1'] = tmp['seller_name_1']
    data['seller_name_2'] = tmp['seller_name_2']
    data['seller_name_3'] = tmp['seller_name_3']
    data['seller_price_1'] = tmp['seller_price_1']
    data['seller_price_2'] = tmp['seller_price_2']
    data['seller_price_3'] = tmp['seller_price_3']
    data = data.drop('sellers', axis=1)
    data['seller_price_1'] = pd.to_numeric(data['seller_price_1'], downcast='float', errors='coerce')
    data['seller_price_2'] = pd.to_numeric(data['seller_price_2'], downcast='float', errors='coerce')
    data['seller_price_3'] = pd.to_numeric(data['seller_price_3'], downcast='float', errors='coerce')

    for i in data:
        random_sample = data[i].dropna().sample(data[i].isnull().sum())
        random_sample.index = data[data[i].isnull()].index
        data.loc[data[i].isnull(), i] = random_sample
    print(data.isna().sum())
    return data


def Feature_Encoder(X, col):
    for c in col:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X
