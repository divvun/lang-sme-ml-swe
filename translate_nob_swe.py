import xml.etree.ElementTree as ET
from google.cloud import translate_v2 as translate

#load  North Sámi - Norwegian Corpus
tree = ET.parse('corpora/nobsme.tmx')
root = tree.getroot()
translate_client = translate.Client()

def translate_text(target, text, model="nmt"):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    #print(u"Text: {}".format(result["input"]))
    #print(u"Translation: {}".format(result["translatedText"]))
    #print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result["translatedText"]

for i, child in enumerate(root):
    if i==1:
        for e, elem in enumerate(child):
            if e%1000 == 0:
                print(e, "/", len(child), ':', (e/len(child))*100, "%")
            for subelem in elem:
                lang = subelem.get('{http://www.w3.org/XML/1998/namespace}lang')
                
                if lang == "nob":
                    for seg in subelem: 
                        sentence = seg.text
                        translation = translate_text('sv', sentence) # get translation
                        seg.text = translation # replace norwegian sentence in xml tree
                    subelem.set('{http://www.w3.org/XML/1998/namespace}lang', 'sv') # set language to swedish in tree  
                elif lang == None:
                    lang2 = subelem.attrib['lang']
                    if lang2 == "nob":
                        for seg in subelem: 
                            sentence = seg.text
                            translation = translate_text('sv', sentence) # get translation
                            seg.text = translation # replace norwegian sentence in xml tree
                        subelem.set('lang', 'sv') # set language to swedish in tree  

                    
tree.write('corpora/smeswebig.tmx', encoding='utf-8') #save translations to new file




