import pandas as pd
import asyncio


from soundcloud_scraper import mus_search



def mus_collect(url):

    input = asyncio.run(mus_search(url))

    data = pd.DataFrame.from_records(input)

    df = pd.DataFrame(data)

    df.to_csv('data\scdb.csv', mode='a', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    mus_collect('https://soundcloud.com/user-818400639-218240550/sets/zzgcthqyvuhe?si=bfbda8427ad8474ebad3fad2379462b1&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing')