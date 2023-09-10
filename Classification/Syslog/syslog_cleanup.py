import re
import pandas as pd


# Specify the path to your syslog file
syslog_file_path = './data/Syslog/syslog'
filtered_syslog_file_path = './data/Syslog/syslog.cvs'


def check_for_exclusion_pattern(line):
    keywords_pattern = r'"error": "none"'
    keyword_match = re.search(
        keywords_pattern, line, re.IGNORECASE)
    if keyword_match:
        return True
    else:
        return False


def convert_syslog_to_dataframe(filepath):
    # Define a regular expression pattern to extract the information
    # \w{3} matches exactly three word characters (letters or digits). This corresponds to the month abbreviation like "Jul."
    # \d{2}:\d{2}:\d{2} matches the time in the format "HH:MM:SS," where HH represents the hours, MM represents the minutes, and SS represents the seconds.
    # (\S+): This part of the regular expression captures the next non-space sequence of characters, which corresponds to the application name (e.g., "app.py" in your syslog lines).
    # \S+ matches one or more non-space characters.
    pattern = r'(\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\S+) (.*)'

    # Create empty lists to store extracted data
    date_time_list = []
    application_list = []
    detail_list = []
    labels_list = []
    records_processed = 2
    # Extract data from syslog lines
    with open(syslog_file_path, 'r') as file:
        for line in file:
            if(records_processed == 975):
                print("Found target")
            line = line.rstrip()
            match = re.match(pattern, line)
            if match:
                keywords_pattern = r'error|failed|failure|ERRCONNECT|\berr\b'
                keyword_match = re.search(
                    keywords_pattern, line, re.IGNORECASE)
                if keyword_match:
                    print(
                        f"The string on line {records_processed} contains the target keyword. {line}")
                    if(check_for_exclusion_pattern(line) == True):
                        label = 0
                    else:
                        label = 1
                else:
                    label = 0
                date_time = match.group(1)
                application = match.group(3)
                detail = match.group(4)
                date_time_list.append(date_time)

                if "[OSD]" in detail:
                    application = "OSD"
                    osd_pattern = r'\[OSD\] : \[.+?\] \[.+?\] - (.*)'
                    match = re.match(osd_pattern, detail)
                    num_groups = len(match.groups())
                    if match and (num_groups == 1):
                        detail_ammended = match.group(1)
                        detail_list.append(detail_ammended)
                    else:
                        print(
                            f"something went wrong on line {records_processed}: {detail}")
                        detail_list.append(detail)
                else:
                    detail_list.append(detail)
                application_list.append(application)
                labels_list.append(label)
                records_processed = records_processed + 1
            else:
                print(
                    f"Could not match or filter this line: [{line}] on line number {records_processed}")
    # Create a DataFrame
    print(f"Processed {records_processed} records")
    df = pd.DataFrame({
        "Date/Time": date_time_list,
        "Application": application_list,
        "Detail": detail_list,
        "Label": labels_list
    })
    return df


# def quick_test():
#     "Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[154037]: Program parameters:"


# Display the DataFrame
# print(df)
df = convert_syslog_to_dataframe(syslog_file_path)
df.to_csv(filtered_syslog_file_path, index=False)
# df2 = pd.read_csv(filtered_syslog_file_path)
# for value in df2["Detail"]:
#     print(f"Detail only: {value}")
