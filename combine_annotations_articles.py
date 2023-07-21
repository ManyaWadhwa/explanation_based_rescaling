import json
import pandas

def convert_json_to_csv(df, all_articles, annotations):
    """
    Given all articles and annotations; this function combines them are returns a dataframe
    where each row is one annotation
    :param df:
    :param all_articles:
    :param annotations:
    :return:
    """
    for article_num in annotations:
        question_details = annotations[article_num]['question_details']
        article_details = all_articles[article_num]
        article = article_details['full_article']
        for questionid in question_details:
            question = question_details[questionid]['question']
            for system in question_details[questionid]['system']:
                answer = question_details[questionid]['system'][system]['answer']
                annotation_details = question_details[questionid]['system'][system]['annotation_details']
                for annotationid in annotation_details:
                    annotation = annotation_details[annotationid]
                    correct  =annotation['correct_label']
                    complete = annotation['complete_label']
                    feedback = annotation['feedback']
                    ebr_raw = annotation['ebr_raw']
                    ebr_num = annotation['ebr_num']
                    workerid = annotation['workerid']
                    df['article_num'].append(article_num)
                    df['article'].append(article)
                    df['question_id'].append(questionid)
                    df['question'].append(question)
                    df['qa_system'].append(system)
                    df['answer'].append(answer)
                    df['complete'].append(complete)
                    df['correct'].append(correct)
                    df['feedback'].append(feedback)
                    df['ebr_raw'].append(ebr_raw)
                    df['ebr_num'].append(ebr_num)
                    df['workerid'].append(workerid)
    return df

def main():
    """
    reads the inq and inq-extended articles and annotations; combines them into one unrolled csv
    where each row is an annotation from the user
    :return: pandas dataframe
    """
    inq_articles = json.load(open("inq_articles.json"))
    inq_ext_articles = json.load(open("inq_extended_articles.json"))

    inq_annotations = json.load(open("inq_annotations.json"))
    inq_extended_annotations = json.load(open("inq_extended_annotations.json"))

    # combine things
    df ={
        "article_num":[],
        "article":[],
        "question_id":[],
        "question":[],
        "qa_system":[],
        "answer":[],
        "complete":[],
        "correct":[],
        "feedback":[],
        "ebr_raw":[],
        "ebr_num":[],
        "workerid":[]
    }

    df = convert_json_to_csv(df, inq_articles, inq_annotations)
    df = convert_json_to_csv(df, inq_ext_articles, inq_extended_annotations)
    data = pandas.DataFrame(df)
    return data
if __name__=="__main__":
    data = main()
