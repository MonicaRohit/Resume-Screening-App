
import streamlit as st
import pickle
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words= 'english')

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


def cleanResume(txt):
    cleanTxt = re.sub('http\S+\s', ' ', txt)
    cleanTxt = re.sub('RT|cc', ' ', cleanTxt)
    cleanTxt = re.sub('#\S+\s', ' ', cleanTxt)
    cleanTxt = re.sub('@\S+', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)

    return cleanTxt

# Web App
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume', type= ['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "Devops Engineer",
            20: "Python Developer",
            6: "Data Science",
            12: "HR",
            0: "Advocate",
            1: "Arts",
            24: "Web Designing",
            16: "Mechanical Engineer",
            22: "Sales",
            14: "Health and fitness",
            5: "Civil Engineer",
            4: "Business Analyst",
            21: "SAP Developer",
            2: "Automation Testing",
            11: "Electrical Engineering",
            18: "Operations Manager",
            17: "Network Security Engineer",
            19: "PMO",
            7: "Database",
            13: "Hadoop",
            10: "ETL Developer",
            9: "DotNet Developer",
            3: "Blockchain"
        }

        category_name = category_mapping.get(prediction_id, "unknown")

        st.write("Predicted Category:", category_name)


# python main
if __name__ == "__main__":
    main()