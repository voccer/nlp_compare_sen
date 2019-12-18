
import re
import string


def normalize_Text(comment):

    comment = comment.encode().decode()
    comment = comment.lower()
    words = comment.split()
    pre_text= []
    for word in words:
        if(len(word) <18):
            pre_text.append(word)
    comment1 = ' '.join(pre_text)
    # thay gia tien bang text
    moneytag = [u'k', u'đ', u'ngàn', u'nghìn', u'usd', u'tr', u'củ', u'triệu', u'yên']

    for money in moneytag:
        comment1 = re.sub('(^\d*([,.]?\d+)+\s*' + money + ')|(' + '\s\d*([,.]?\d+)+\s*' + money + ')', ' monney ',
                         comment1)

    comment1 = re.sub('(^\d+\s*\$)|(\s\d+\s*\$)', ' monney ', comment1)
    comment1 = re.sub('(^\$\d+\s*)|(\s\$\d+\s*\$)', ' monney ', comment1)
    comment1 = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', comment1)
    comment1 = re.sub(r'[\w\.-]+@[\w\.-]+', ' EMAIL ', comment1)
    comment1 = re.sub(r'[\d]+%', "PERCENT", comment1)
    comment1 = re.sub(r'[\d]+[,.][\d]+', 'DIGIT', comment1)


    # loai dau cau: nhuoc diem bi vo cau truc: vd; km/h. V-NAND
    listpunctuation = string.punctuation
    for i in listpunctuation:
        comment1 = comment1.replace(i, ' ')

    # thay thong so bang specifications
    comment1 = re.sub('^(\d+[a-z]+)([a-z]*\d*)*\s|\s\d+[a-z]+([a-z]*\d*)*\s|\s(\d+[a-z]+)([a-z]*\d*)*$', ' ', comment1)
    comment1 = re.sub('^([a-z]+\d+)([a-z]*\d*)*\s|\s[a-z]+\d+([a-z]*\d*)*\s|\s([a-z]+\d+)([a-z]*\d*)*$', ' ', comment1)

    # thay thong so bang text lan 2
    comment1 = re.sub('^(\d+[a-z]+)([a-z]*\d*)*\s|\s\d+[a-z]+([a-z]*\d*)*\s|\s(\d+[a-z]+)([a-z]*\d*)*$', ' ', comment1)
    comment1 = re.sub('^([a-z]+\d+)([a-z]*\d*)*\s|\s[a-z]+\d+([a-z]*\d*)*\s|\s([a-z]+\d+)([a-z]*\d*)*$', ' ', comment1)

    # xu ly lay am tiet
    comment1 = re.sub(r'(\D)\1+', r'\1', comment1)

    return comment1



