import pandas as pd

import utilities

data_file = "./data/DataScientist.csv"

df = pd.read_csv(data_file, encoding='utf-8')

emails = utilities.get_emails_from_dataframe(df, 'Job Description')
print(emails)
urls = utilities.get_urls_from_dataframe(df, 'Job Description')
print(urls)


def get_closest_email_lev(df, email):
    df = utilities.find_levenshtein_dataframe(email, df, "emails")
    column_name = 'distance_to_' + email
    minimum_value_email_index = df[column_name].idxmin()
    email = df.loc[minimum_value_email_index]['emails']
    return email


def get_closest_email_jaro(df, email):
    df = utilities.find_jaro_dataframe(email, df, "emails")
    column_name = 'distance_to_' + email
    maximum_value_email_index = df[column_name].idxmax()
    email = df.loc[maximum_value_email_index]['emails']
    return email


def main():
    df = pd.read_csv(data_file, encoding='utf-8')
    emails = utilities.get_emails_from_dataframe(df, 'Job Description')
    new_df = pd.DataFrame(emails, columns=['emails'])
    input_string = "rohitt.macdonald@prelim.com"
    email = get_closest_email_jaro(new_df, input_string)
    print(email)


if (__name__ == "__main__"):
    main()
