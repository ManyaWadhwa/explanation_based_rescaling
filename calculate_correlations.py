import json
import pandas
from scipy.stats import kendalltau

def convert_json_to_csv(df, all_articles, annotations):
    for article_num in annotations:
        # print(article_num)
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
        where each row is an annotation from the user and calculates the pairwise kendall tau
        pre and post rescaling
        :return:
    """
    inq_articles = json.load(open("data/inq_articles.json"))
    inq_ext_articles = json.load(open("data/inq_extended_articles.json"))

    inq_annotations = json.load(open("data/inq_annotations.json"))
    inq_extended_annotations = json.load(open("data/inq_extended_annotations.json"))

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

    ## calculate correlation
    def rescore(row):
        s = row['ebr_num']
        if s == s:
            if row['complete'] == 'complete' and row['correct'] == 'correct':
                return 100
            return s
        return s

    mapping = {"complete": 100, "missing_minor": 70, "missing_major": 30, "missing_all": 0}
    data['ebr_num'] = data.apply(lambda x: rescore(x), axis = 1)
    data['complete_mapping'] = data['complete'].apply(lambda x: mapping[x])
    data['combined'] = data.apply(lambda x: x['ebr_num'] if x['ebr_num']==x['ebr_num'] else x['complete_mapping'], axis = 1)

    unique_workerids = list(set(data['workerid']))
    unique_workerids.sort()
    pairwise_info = {"worker1": [], "worker2": [], "corr": [],"corr_og":[], "len": []}  # get pairwise
    for i in range(len(unique_workerids)):
        w1 = data[data['workerid'] == unique_workerids[i]]
        for j in range(i + 1, len(unique_workerids)):
            w2 = data[data['workerid'] == unique_workerids[j]]
            combined = pandas.merge(w1, w2,
                                    on=['article_num', 'question_id', 'article', 'question','qa_system', 'answer'],
                                    how='inner')
            score_1 = combined['combined_x']
            score_2 = combined['combined_y']
            corr = kendalltau(score_1, score_2).statistic
            corr_og = kendalltau(combined['complete_mapping_x'],combined['complete_mapping_y']).statistic
            pairwise_info['worker1'].append(unique_workerids[i])
            pairwise_info['worker2'].append(unique_workerids[j])
            pairwise_info['corr'].append(corr)
            pairwise_info['corr_og'].append(corr_og)
            pairwise_info['len'].append(len(combined))
    pairwise_info_pd = pandas.DataFrame(pairwise_info)
    corr = pairwise_info_pd['corr'].mean()
    corr_og = pairwise_info_pd['corr_og'].mean()
    print(f"Pairwise correlation with original labels:{corr_og}")
    print(f"Pairwise correlation after rescaling:{corr}")
if __name__=="__main__":
    main()