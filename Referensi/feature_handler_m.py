from __future__ import division
import re
from collections import Counter
from file_handler import loadlist, savelist, loadtxt
import itertools
import math
from gensim.models import Word2Vec
import normalizer
import os
import sent2vec

from inanlp.postagger import POSTagger

class TextFeature:
	bow = []
	term_corpus_pos = []#normalized
	term_corpus_neg = []#normalized
	sentence_corpus_pos = []#normalized
	sentence_corpus_neg = []
	we_model = Word2Vec.load('we_model/model_all_unstem.bin')
	se_model = sent2vec.Sent2vecModel()
	we_wlist = we_model.wv.vocab
	norm = normalizer.normalizer()
	tagger = POSTagger()

	#Initialize bow list and corpus for tf/idf
	def __init__(self):
		neg_corpus = "rude_corpus/norm_negative.txt"
		pos_corpus = "rude_corpus/norm_positive.txt"
		self.se_model.load_model('we_model/s2v_model_corpus20.bin')
		if os.path.exists(neg_corpus):
			self.sentence_corpus_neg = loadlist(neg_corpus)
			text = 	self.norm.clean_nonalphanum(loadtxt(neg_corpus).replace("\n"," "),True)
			self.term_corpus_neg = text.split(" ")

		if os.path.exists(pos_corpus):
			self.sentence_corpus_pos = loadlist(pos_corpus)
			text = 	self.norm.clean_nonalphanum(loadtxt(pos_corpus).replace("\n"," "),True)
			self.term_corpus_pos = text.split(" ")

		if os.path.exists("dict/bow_occ.txt"):
			self.bow = loadlist("dict/bow_occ.txt")
		else:
			self.bow_occ_file(self,pos_corpus,neg_corpus)
	
	#Check if string is number
	def is_number(self,token):
		try:
			float(token)
			return True
		except ValueError:
			return False

	#Load BOW-list from file
	def load_bow_list(self,file):
		self.bow = loadlist(file)

	#Determine major term based on num of occurrence in a text
	#Input	:
	#	- text_corpus	: list of sentence in form of text/string
	#	- trange 		: term's range of rank
	#Output : trange term (sorted) 
	def corpus_major_term_occ(self,text_corpus,trange):
		text_corpus = self.norm.normalize_text(text_corpus)#text_corpus.replace("-"," ").replace("\t"," ").replace("\n"," ").lower()
		text_corpus = re.sub('<.*?>', '', text_corpus)
		#text_corpus = re.sub(r'[^a-zA-Z0-9 ]', '', text_corpus)
		text_corpus = re.sub(r'  ', ' ', text_corpus)
		token_list = text_corpus.split(" ")

		#replace number
		for i in range (0, len(token_list)):
			if self.is_number(token_list[i]):
				token_list[i] = "1" 
		
		token_counter = Counter(token_list)
		return (sorted(token_counter.items(), key=lambda x: x[1],reverse=True))[:trange]

	#Determine bow relevant for corpus c1 (delete relevant major term in c1 if it appeared as c2 major term)
	#Input	:
	#	- major_term_c1	: list of major term in relevant corpus
	#	- major_term_c2	: list of major term in compared corpus
	#	- file 			: output file to save list
	#Outpur: major term relevant to c1 stored in self.bow and external file
	def bow_occ_list(self,major_term_c1,major_term_c2,file="dict/bow_occ.txt"):
		new_list_terms = []
		for i in range (0, len(major_term_c1)):
			mjterm = major_term_c1[i][0]
			if len([row[0] for row in major_term_c2 if row[0]==mjterm])==0:
				new_list_terms.append(mjterm)
		self.bow = new_list_terms
		if file!="":
			savelist(self.bow,file,"")

	#Determine bow relevant for corpus c1 (delete relevant major term in c1 if it appeared as c2 major term)
	#Corpus loaded from file
	#Input	:
	#	- fcorpus1	: filename of relevant corpus
	#	- fcorpus2	: filename of compared corpus
	#	- trange1 	: range of major term in fcorpus1
	#	- trange2 	: range of major term in fcorpus2
	#	- file 			: output file to save list
	#Outpur: major term relevant to c1 stored in self.bow and external file
	def bow_occ_file(self,fcorpus1,fcorpus2,trange1=50,trange2=100,file="dict/bow_occ.txt"):
		major_term1 = self.corpus_major_term_occ(loadtxt(fcorpus1),trange1)
		major_term2 = self.corpus_major_term_occ(loadtxt(fcorpus2),trange2)
		self.bow_occ_list(major_term1,major_term2,file)

	#cek load
	
	def tf_pos(self,term):
		return self.term_corpus_pos.count(term)/len(self.term_corpus_pos)

	def idf_pos(self,term):
		ndt = len([sent for sent in self.sentence_corpus_pos if term in sent])
		if ndt==0:
			return 0
		return math.log(len(self.sentence_corpus_pos)/ndt)

	def tfidf_pos(self,term):
		return self.tf_pos(term)*self.idf_pos(term)
	
	def get_class_id(self,term,nid):
		_class = "0"
		_id = "0"
		t = term
		if "_" in t:
			token = t.split("_")
			_id = nid+"_"+token[0]
			_class = token[1]
			t = token[0]
		else:
			_id = nid+"_"+t
		return _id, t, _class

#-------------------TERM LEVEL--------------------
	def mono_extract_tfidf(self,term):
		return self.tfidf_pos(term)

	def mono_extract_we(self,term):
		term = self.norm.clean_norm_sentence(term)
		token = term.split(" ")
		we = []
		for t in token:
			if t in self.we_wlist:
				we.append(self.we_model[t].tolist())
			else:
				we.append([0]*100)
		return [sum(e)/len(e) for e in zip(*we)]

#-----------------SENTENCE LEVEL-------------------
	def sentence_postag(self,sentence,nid):
		sentence = self.norm.clean_nonalphanum(sentence,True).split(" ")
		ori_norm = []
		norm_wlist = []
		for word in sentence:
			_id, word, _class = self.get_class_id(word,nid)
			normalized = self.norm.norm_formalize(word)
			if " " in normalized:
				normalized = normalized.split(" ")
				for n in normalized:
					ori_norm.append([_id,_class,word,n])
					norm_wlist.append(n)
			else:
				ori_norm.append([_id,_class,word,normalized])
				norm_wlist.append(normalized)
		tag = self.tagger.tag(tokens=norm_wlist)

		pos_list = list(self.tagger.coarse_mapping)
		size = len(pos_list)
		#print(size)
		feature=[]
		idx = 0
		#print(sentence)
		for word in sentence:
			word = word.split("_")[0]
			#print(word)
			if idx <len(ori_norm):
				feature_word = [0]*size
				while idx <len(ori_norm) and word==ori_norm[idx][2]:
					word_tag = tag[idx].tag
					#print(tag[idx])
					feature_word[pos_list.index(word_tag)]=1
					idx+=1
					if idx<len(ori_norm) and idx>0:
						if ori_norm[idx-1][3]==ori_norm[idx][3] and ori_norm[idx-1][2]==ori_norm[idx][2]:
							word += "_"

			feature.append([ori_norm[idx-1][0]]+feature_word+[ori_norm[idx-1][1]])
		return(feature)		

	def sentence_tfidf(self, sentence,nid):
		sentence = self.norm.clean_nonalphanum(sentence,True).split(" ")
		feature = []
		#sentence_clean = self.norm.clean_norm_sentence(sentence)
		for word in sentence:
			_id, word, _class = self.get_class_id(word,nid)
			t = self.norm.norm_formalize(word)
			feature.append([_id,self.mono_extract_tfidf(t),_class])
		return feature

	def sentence_bow_occ(self,sentence,nid):
		sentence = self.norm.clean_nonalphanum(sentence,True)
		sentence_clean = self.norm.clean_norm_sentence(sentence)

		feature_final = []#self.get_class(sentence.split(" "),nid)
		feature_bow = []
		term_list = sentence_clean.split(" ")
		for term in self.bow:
			feature_bow.append(term_list.count(term))

		term = sentence.split(" ")
		for t in term:
			_id, t, _class = self.get_class_id(t,nid)
			feature_final.append([_id]+feature_bow+[_class])
		return feature_final
	
	def sentence_w2v(self,sentence,nid,gram,pos):
	#pos -2: t as last -1: t as center; 0: t as first
		feature = []
		#preprocess
		sentence = self.norm.clean_nonalphanum(sentence,True)
		terms = sentence.split(" ")
		
		for t in terms:
			feature_t = []
			_class	= ""
			_id, t, _class = self.get_class_id(t,nid)
			feature_t.append(_id)

			feature_t+=self.mono_extract_we(t)
			feature_t.append("class"+_class)
			feature.append(feature_t)

		feature_len = len(feature[0][1:-1])
		#for n gram
		if gram>1:
			final_feature = []
			for term in range(0,len(feature)):
				feature_n = [feature[term][0]]	#get id
				curr_id = int(term+(gram-1)/2*pos) 		#set start idx
				#get n gram feature
				for i in range (0,gram):
					if curr_id>=0 and curr_id<len(feature):
						feature_n = feature_n+feature[curr_id][1:-1]
					else:
						feature_n = feature_n+[0]*feature_len
					curr_id += 1 #term's ids
				final_feature.append(feature_n+[feature[term][-1]])
				del feature_n[:]
			return final_feature
		return feature
	def count_we_avg(self,we):
		res = []
		for y in range (0,len(we[0])):
			_sum = 0
			for x in range (0,len(we)):
				#print(we[x][y])
				_sum+= float(we[x][y])
			res.append(_sum/(len(we)))
		#print(len(res))
		return res

	def sentence_s2v_avg(self,sentence,nid,split=False):
		feature = []
		#preprocess
		sentence = self.norm.clean_nonalphanum(sentence,True)
		terms = sentence.split(" ")
		
		for t in terms:
			feature_t = []
			_class	= ""
			_id, t, _class = self.get_class_id(t,nid)
			feature_t.append(_id)

			feature_t+=self.mono_extract_we(t)
			feature_t.append("class"+_class)
			feature.append(feature_t)
		#print(feature)
		value = [f[1:-1] for f in feature]
		for term in range (0,len(feature)):
			
			if not split:
				feature[term] = [feature[term][0]]+self.count_we_avg(value)+[feature[term][-1]]
			else:
				if term == 0:
					feature[term] = [feature[term][0]]+[0]*100+value[term]+self.count_we_avg(value[term:])+[feature[term][-1]]
				elif term == len(feature)-1:
					feature[term] = [feature[term][0]]+self.count_we_avg(value[:term])+value[term]+[0]*100+[feature[term][-1]]
				else:
					feature[term] = [feature[term][0]]+self.count_we_avg(value[:term])+value[term]+self.count_we_avg(value[term:])+[feature[term][-1]]
		return feature

	def sentence_s2v_model(self,sentence,nid,split=False):
		feature = []
		#preprocess
		sentence = self.norm.clean_nonalphanum(sentence,True)
		terms = sentence.split(" ")
		feature_t =[]
		for t in terms:
			_id, t, _class = self.get_class_id(t,nid)
			t = self.norm.norm_formalize(t)
			feature_t.append([_id]+[t]+["class"+_class])
		sentence_norm=""
		if split==False:
			for s in feature_t:
				sentence_norm += s[1]+" "
			sent_emb = (self.se_model.embed_sentence(sentence_norm[:-1])).tolist()
			for i in range (0,len(feature_t)):
				feature.append([feature_t[i][0]]+sent_emb+[feature_t[i][-1]])
		else:
			for i in range (0,len(feature_t)):
				sent_bef = [0]*100
				if i>0:
					sent_bef = ""
					for ib in range(0,i):
						sent_bef += feature_t[ib][1]+" "
					sent_bef = self.se_model.embed_sentence(sent_bef[:-1]).tolist()
				sent_af = [0]*100
				if i<len(feature_t)-1:
					sent_af = ""
					for ia in range(i+1,len(feature_t)):
						sent_af+= feature_t[ia][1]+" "
					sent_af = self.se_model.embed_sentence(sent_af[:-1]).tolist()
				curr_we = self.mono_extract_we(feature_t[i][1])
				feature.append([feature_t[i][0]]+sent_bef+curr_we+sent_af+[feature_t[i][-1]])
		return feature



#-------OTHER------------------
	#initialize feature header
	def init_header(self,size,name):
		header = ['ID-Name']
		for i in range(1,size+1):
			header.append(name+str(i))
		header.append('class')
		return header

	def merge_features(self,list_feature):
		final_feature = []
		for term in range (0,len(list_feature[0])):
			feature = [list_feature[0][term][0]]#get id
			for f in list_feature:
				feature+= f[term][1:-1]#get values
			feature += [list_feature[0][term][-1]]#class
			final_feature.append(feature)
		return final_feature

	def extract_file(self,file,output_file):
		corpus_all = loadlist(file,"\t")
		corpus_id = [nid[0] for nid in corpus_all]
		corpus_sent = [nid[1] for nid in corpus_all]
		feature = []

		we = []
		bow_occ = []
		tfidf = []
		pos = []
		s2v_all = []
		s2v_split = []

		for s in range (0,len(corpus_id)):
			self.sentence_postag(corpus_sent[s],corpus_id[s])
			we+=(self.sentence_w2v(corpus_sent[s],corpus_id[s],1,-2))
			s2v_all+= self.sentence_s2v_model(corpus_sent[s],corpus_id[s],False)
			s2v_split+= self.sentence_s2v_model(corpus_sent[s],corpus_id[s],True)
			#s2v_all+= self.sentence_s2v_avg(corpus_sent[s],corpus_id[s],False)
			#s2v_split+= self.sentence_s2v_avg(corpus_sent[s],corpus_id[s],True)
			bow_occ+=(self.sentence_bow_occ(corpus_sent[s],corpus_id[s]))
			tfidf+=(self.sentence_tfidf(corpus_sent[s],corpus_id[s]))
			pos+=(self.sentence_postag(corpus_sent[s],corpus_id[s]))
		
		we = [self.init_header(len(we[0][1:-1]),"we")]+we
		s2v_all = [self.init_header(len(s2v_all[0][1:-1]),"s2v_all")]+s2v_all
		s2v_split = [self.init_header(len(s2v_split[0][1:-1]),"s2v_split")]+s2v_split
		bow_occ = [self.init_header(len(bow_occ[0][1:-1]),"bow")]+bow_occ
		tfidf = [self.init_header(len(tfidf[0][1:-1]),"tfidf")]+tfidf
		pos = [self.init_header(len(pos[0][1:-1]),"pos")]+pos

		feature = self.merge_features([pos,tfidf,s2v_split])
		savelist(feature,output_file,",")

t = TextFeature()
t.bow_occ_file("rude_corpus/raw_positive.txt","rude_corpus/raw_negative.txt",100,150,"dict/bow_occ.txt")
t.extract_file("rude_corpus/train_corpus.txt","train_P-TF-SSM_uns.csv")
t.extract_file("rude_corpus/test_corpus.txt","test_P-TF-SSM_uns.csv")
'''
corpus_all = loadlist("rude_corpus\\train_corpus.txt","\t")
corpus_id = [nid[0] for nid in corpus_all]
corpus_sent = [nid[1] for nid in corpus_all]
feature = []
we = []
bow_occ = []
tfidf = []

for s in range (0,len(corpus_id)):
	we+=(t.sentence_w2v(corpus_sent[s],corpus_id[s],1,-2))
	bow_occ+=(t.sentence_bow_occ(corpus_sent[s],corpus_id[s]))
	tfidf+=(t.sentence_tfidf(corpus_sent[s],corpus_id[s]))

we = [t.init_header(len(we[0][1:-1]),"we")]+we
bow_occ = [t.init_header(len(bow_occ[0][1:-1]),"bow")]+bow_occ
tfidf = [t.init_header(len(tfidf[0][1:-1]),"tfidf_pos")]+tfidf

feature = t.merge_features([we,bow_occ,tfidf])
savelist(feature,"train_FWUAO_BCT-Us.csv",",")

##---TEST SET
corpus_all = loadlist("rude_corpus\\test_corpus.txt","\t")
corpus_id = [nid[0] for nid in corpus_all]
corpus_sent = [nid[1] for nid in corpus_all]
feature = []
we = []
bow_occ = []

for s in range (0,len(corpus_id)):
	we+=(t.sentence_w2v(corpus_sent[s],corpus_id[s],1,-2))
	bow_occ+=(t.sentence_bow_occ(corpus_sent[s],corpus_id[s]))
	tfidf+=(t.sentence_tfidf(corpus_sent[s],corpus_id[s]))

we = [t.init_header(len(we[0][1:-1]),"we")]+we
bow_occ = [t.init_header(len(bow_occ[0][1:-1]),"bow")]+bow_occ
tfidf = [t.init_header(len(tfidf[0][1:-1]),"tfidf_pos")]

feature = t.merge_features([we,bow_occ,tfidf])
savelist(feature,"test_FWUAO_BCT-Us.csv",",")

#F: feature
#we	: word emb
#U: unigram
#US: unstem
#A: All
#O: normalizer original
'''