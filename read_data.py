import xml.etree.ElementTree as ET


def read_data(xml_file):
    swedish_sent = []
    sami_sent = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for i, child in enumerate(root):
        if i == 1:
            for i, elem in enumerate(child):
                for e, subelem in enumerate(elem):
                    lang = subelem.get('{http://www.w3.org/XML/1998/namespace}lang')
                    if lang == "sv":
                        for seg in subelem:
                            sentencesv = seg.text
                    elif lang == "sme":
                        for seg in subelem:
                            sentencesme = seg.text
                    elif lang == None:
                        lang = subelem.attrib['lang']
                        if lang == "sv":
                            for seg in subelem:
                                sentencesv = seg.text                                
                        elif lang == "sme":
                            for seg in subelem:
                                sentencesme = seg.text
                    if e%2 != 0:
                        if sentencesv != None:
                            if sentencesme != None:
                                sami_sent.append(sentencesme)
                                swedish_sent.append(sentencesv)
    return sami_sent, swedish_sent

