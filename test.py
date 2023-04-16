import pandas as pd
from datetime import datetime

columns = ['comment_text', 'output']

def test(model, data): 
    prediction = list(map(lambda arr: arr[0], model.predict(data)))
    df = pd.DataFrame(data=list(zip(data, prediction)) ,columns=columns)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f'results/result-{now}.csv'
    df.to_csv(filename, index=True)