import pandas as pd
from datetime import datetime

columns = ['comment_text', 'output']
treshold = 0.51

def test(model, data): 
    prediction = list(map(lambda arr: arr[0], model.predict(data)))
    df = pd.DataFrame(data=list(zip(data, prediction)) ,columns=columns)

    toxic = df[df['output'] >= treshold]
    clean = df[df['output'] < treshold]

    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    toxic.to_csv(f'results/toxic-{now}.csv', index=True)
    clean.to_csv(f'results/clean-{now}.csv', index=True)
