//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

//modification begin
#define MAX_MAP_STRING 300
#define MAX_MORPHEME_SIZE 100
//modification end

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary (threshold)

typedef float real;                    // Precision of float numbers

//modification begin
struct pos{
  long long position;
  float weight;
};

struct word_map{
  char *word;
  int pn, rn, sn;
  struct pos *prefix;
  struct pos *root;
  struct pos *suffix;
};
//modification end

struct vocab_word {
  long long cn; // word count
  int *point;
  char *word, *code, codelen;
  //modification begin
  int pn, rn, sn;
  struct pos *prefix;
  struct pos *root;
  struct pos *suffix;
  //modification end
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
//modification begin
char wordmap_file[MAX_STRING];
long long map_size = 0;
struct word_map *wordMap;
int *map_hash;
//modification end
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, dim = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable; //syn0: word vector; syn1: parameter vector; syn1neg: parameter vector for negative sampling
clock_t start;

int hs = 0, negative = 5;

const int table_size = 1e8;
int *table;

void InitUnigramTable() { // init the negative sampling map table in terms of the frequencies of words
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power); // total count of all words
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow; // calculate the frequency of each word
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0; ////ASCII '\0'
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash]; //find and return the word
    hash = (hash + 1) % vocab_hash_size; // continue searching, forward direction
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;

  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;

  //modification begin
  vocab[vocab_size].prefix = NULL;
  vocab[vocab_size].root = NULL;
  vocab[vocab_size].suffix = NULL;
  vocab[vocab_size].pn = 0;
  vocab[vocab_size].rn = 0;
  vocab[vocab_size].sn = 0;
  //modification end

  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}


// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; ////init hash table, default -1
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);

    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else {
    free(vocab[a].word);
  }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

//modification begin
int SearchMap(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (map_hash[hash] == -1) return -1;
    if (!strcmp(word, wordMap[map_hash[hash]].word)) return map_hash[hash]; //find and return the word position in wordMap
    hash = (hash + 1) % vocab_hash_size; // continue searching, forward direction
  }
  return -1;
}

char* StrReplace(char *str, const char *oldStr, const char *newStr){
  char *ptr = NULL;
  while(ptr = strstr(str, oldStr))
  {
    memmove(ptr + strlen(newStr), ptr + strlen(oldStr), strlen(ptr) - strlen(oldStr) + 1);
    memcpy(ptr, &newStr[0], strlen(newStr));
  }
  return str;
}

char** SplitStr(char *str, char spliter, int *cnt)
{
  char ** dst = NULL;
  char *ptr1, *ptr2;
  int idx;
  *cnt = 0;
  
  ptr2 = str;
  ptr1 = strchr(ptr2, spliter);

  while (1)
  {
    if(ptr1 == NULL){
      if(strlen(ptr2) <= 0)
        break;
      else{
        *cnt += 1; break;
      }
    }
    if(ptr1 != ptr2)
    {
      *cnt += 1;
    }
    ptr2 = ptr1 + 1;
    ptr1 = strchr(ptr2, spliter);
  }

  dst = (char**) malloc (*cnt * sizeof(char*));

  idx = 0;
  ptr2 = str;
  ptr1 = strchr(ptr2, spliter);
  while (1)
  {
    if(ptr1 == NULL){
      if(strlen(ptr2) <= 0)
        break;
      else{
        dst[idx] = (char*)calloc(strlen(ptr2) + 1, sizeof(char));
        strncpy(dst[idx], ptr2, strlen(ptr2));
        break;
      }
    }
    if(ptr1 != ptr2)
    {
      dst[idx] = (char*)calloc(ptr1 - ptr2 + 1, sizeof(char));
      strncpy(dst[idx], ptr2, ptr1 - ptr2);
      idx++;
    }
    ptr2 = ptr1 + 1;
    ptr1 = strchr(ptr2, spliter);
  }
  
  return dst;
}

char* GetMainWordOfPhrase(char *str, char spliter){
  int idx, maxIdx;
  int len;
  char *dst;
  int cnt;
  char ** tmp = SplitStr(str, spliter, &cnt);
  if(cnt == 0) return NULL;
  maxIdx = 0;
  len = strlen(tmp[0]);
  for(idx = 0; idx < cnt; idx++){
    if(len <= strlen(tmp[idx])){
      maxIdx = idx;
      len = strlen(tmp[maxIdx]);
    }
  }
  dst = (char *)calloc(strlen(tmp[maxIdx]) + 1, sizeof(char));
  strcpy(dst, tmp[maxIdx]);
  
  for(idx = 0; idx < cnt; idx++) free(tmp[idx]);
  free(tmp);
  return dst;
}

void LoadMapData(){
  long long vIdx, curIdx;
  long long idx;
  unsigned int hash;
  //int strLen;
  char str[MAX_MAP_STRING];

  char spliter = '#';
  char spliter2 = ',';
  char spliter3 = ' ';

  FILE *fmap;

  fmap = fopen(wordmap_file, "rb");
  if (fmap == NULL) {
    printf("ERROR: wordmap file not found!\n");
    exit(1);
  }

  fseek(fmap, 0, SEEK_SET);
  map_size = 0;
  while(1){
    if(NULL == fgets(str, MAX_MAP_STRING, fmap)) break;
    char *ptr;
    ptr = strchr(str, spliter);
    if(ptr - str + 1 <= 1)continue;//remove useless lines
	  map_size++;
  }

  wordMap = (struct word_map*)malloc(map_size * sizeof(struct word_map));
  for(idx = 0; idx < map_size; idx++){
    wordMap[idx].word = NULL;
    wordMap[idx].pn = 0;
    wordMap[idx].rn = 0;
    wordMap[idx].sn = 0;
    wordMap[idx].prefix = NULL;
    wordMap[idx].root= NULL;
    wordMap[idx].suffix = NULL;
  }

  map_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  for (idx = 0; idx < vocab_hash_size; idx++) map_hash[idx] = -1; ////init map_hash table, default -1

  fseek(fmap, 0, SEEK_SET);
  map_size = 0;
  while(1){
    if(NULL == fgets(str, MAX_MAP_STRING, fmap)) break;
    StrReplace(str, "\r\n","");
    //printf("[Debug] str = %s\n", str);
    char **mPtr = NULL;
    int mCnt;
    mPtr = SplitStr(str, spliter, &mCnt);
    //printf("[Debug] mCnt = %d\n", mCnt);
    if(mCnt == 0) continue;
    if(mCnt < 4){
      for(idx = 0; idx < mCnt; idx++)
        free(mPtr[idx]);
      free(mPtr); mPtr = NULL;
      continue;
    }
    
    curIdx= SearchMap(mPtr[0]);
    if(curIdx != -1){ // if the word exists in wordMap, then continue
      for(idx = 0; idx < mCnt; idx++)
        free(mPtr[idx]);
      free(mPtr); mPtr = NULL;
      continue;
    }

    curIdx = SearchVocab(mPtr[0]);
    if(curIdx == -1 || curIdx == 0){ // if the word doesn't exist in vocab, then continue
      for(idx = 0; idx < mCnt; idx++)
        free(mPtr[idx]);
      free(mPtr); mPtr = NULL;
      continue; 
    }

    int wordLen = strlen(mPtr[0]);

    wordMap[map_size].word = (char*)calloc(wordLen + 1, sizeof(char));
    strcpy(wordMap[map_size].word, mPtr[0]); //store the target word

    char **subPtr = NULL;
    int wordCnt, effCnt;

    //prefix
    if(0 != strcmp(mPtr[1], " ")){
      subPtr = SplitStr(mPtr[1], spliter2, &wordCnt);

      if(wordCnt > 0){
        wordMap[map_size].prefix = (struct pos*)malloc(wordCnt * sizeof(struct pos)); //store prefix
        effCnt = 0;
        for(idx = 0; idx < wordCnt; idx++)
        {
          char *tmpMainWord = GetMainWordOfPhrase(subPtr[idx], spliter3);
          long long prefixWord = SearchVocab(tmpMainWord);
          if(prefixWord != -1 && prefixWord != 0){
            wordMap[map_size].prefix[effCnt].position = prefixWord;
            wordMap[map_size].prefix[effCnt].weight = 1;
            effCnt++;
          }
        }
        wordMap[map_size].pn = effCnt;

        for(idx = 0; idx < wordCnt; idx++){
          free(subPtr[idx]);
        }
        free(subPtr);subPtr = NULL;
      }
    }

    //root
    if(0 != strcmp(mPtr[2], " ")){
      subPtr = SplitStr(mPtr[2], spliter2, &wordCnt);

      if(wordCnt > 0){
        wordMap[map_size].root = (struct pos*)malloc(wordCnt * sizeof(struct pos)); //store root
        effCnt = 0;
        for(idx = 0; idx < wordCnt; idx++)
        {
          char *tmpMainWord = GetMainWordOfPhrase(subPtr[idx], spliter3);
          long long rootWord = SearchVocab(tmpMainWord);
          if(rootWord != -1 && rootWord != 0){
            wordMap[map_size].root[effCnt].position = rootWord;
            wordMap[map_size].root[effCnt].weight = 1;
            effCnt++;
          }
        }
        wordMap[map_size].rn = effCnt;

        for(idx = 0; idx < wordCnt; idx++){
          free(subPtr[idx]);
        }
        free(subPtr);subPtr = NULL;
      } 
    }
    //suffix
    if(0 != strcmp(mPtr[3], " ")){
      subPtr = SplitStr(mPtr[3], spliter2, &wordCnt);

      if(wordCnt > 0){
        wordMap[map_size].suffix = (struct pos*)malloc(wordCnt * sizeof(struct pos)); //store suffix
        effCnt = 0;
        for(idx = 0; idx < wordCnt; idx++)
        {
          char *tmpMainWord = GetMainWordOfPhrase(subPtr[idx], spliter3);
          long long suffixWord = SearchVocab(tmpMainWord);
          if(suffixWord != -1 && suffixWord != 0){
            wordMap[map_size].suffix[effCnt].position = suffixWord;
            wordMap[map_size].suffix[effCnt].weight = 1;
            effCnt++;
          }
        }
        wordMap[map_size].sn = effCnt;

        for(idx = 0; idx < wordCnt; idx++){
          free(subPtr[idx]);
        }
        free(subPtr);subPtr = NULL;
      } 
    }

    hash = GetWordHash(wordMap[map_size].word);
    while (map_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    map_hash[hash] = map_size;
    map_size++;

    for(idx = 0; idx < mCnt; idx++)
      free(mPtr[idx]);
    free(mPtr); mPtr = NULL;
  }

  fclose(fmap);
  //FreeRefData();
  /*printf("wordMap[%lld].word = %s\n", map_size - 1, wordMap[map_size - 1].word);

  for(curIdx = 0; curIdx < wordMap[map_size - 1].pn; curIdx++) 
    printf("wordMap[%lld].prefix[%d] = %s\n", map_size - 1, curIdx, wordMap[map_size - 1].prefix[curIdx]);

  for(curIdx = 0; curIdx < wordMap[map_size - 1].rn; curIdx++) 
    printf("wordMap[%lld].root[%d] = %s\n", map_size - 1, curIdx, wordMap[map_size - 1].root[curIdx]);

  for(curIdx = 0; curIdx < wordMap[map_size - 1].sn; curIdx++) 
    printf("wordMap[%lld].suffix[%d] = %s\n", map_size - 1, curIdx, wordMap[map_size - 1].suffix[curIdx]);
  return;*/

  printf("[Debug] map_size = %lld\n", map_size);
  printf("[Debug] vocab_size = %lld\n", vocab_size);

  for(vIdx = 1; vIdx < vocab_size; vIdx++){
    //printf("[Debug] vocab[vIdx].word = %s\n", vocab[vIdx].word);
    curIdx = SearchMap(vocab[vIdx].word);

    if(curIdx == -1 || curIdx == 0) continue;
    //printf("[Debug] map_size = %lld\n", map_size);
    //printf("[Debug] vocab_size = %lld\n", vocab_size);
    //printf("[Debug] vIdx = %lld\n",vIdx);
    //printf("[Debug] curIdx = %lld\n",curIdx);
    //printf("[Debug] vocab[vIdx].word = %s\n", vocab[vIdx].word);
    //printf("[Debug] wordMap[curIdx].word = %s\n", wordMap[curIdx].word);

    if(wordMap[curIdx].pn != 0){
      vocab[vIdx].pn = wordMap[curIdx].pn;
      vocab[vIdx].prefix = wordMap[curIdx].prefix;
    }

    if(wordMap[curIdx].rn != 0){
      vocab[vIdx].rn = wordMap[curIdx].rn;
      vocab[vIdx].root = wordMap[curIdx].root;
    }

    if(wordMap[curIdx].sn != 0){
      vocab[vIdx].sn = wordMap[curIdx].sn;
      vocab[vIdx].suffix = wordMap[curIdx].suffix;
    }

    printf("[Debug] Loading %.2f%%%c", (float)vIdx / vocab_size * 100, 13);
    fflush(stdout);
  }
}
//modification end

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }  

  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;

    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }

    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else{ 
			vocab[i].cn++; 
		}

    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab(); //if vocab_size > 0.7*vocab_hash_size, enable the operation of reducing words whose frequencies lower than min_reduce.
  }

  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
	//modification begin
	printf("[Debug] Load word map ...\n");
	LoadMapData();
  printf("[Debug] Load word map successfully!\n");
	//modification end
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * dim * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {// hierachical softmax
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * dim * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < dim; b++)
     syn1[a * dim + b] = 0; // init parameter vector syn1
  }

  if (negative>0) { // negative sampling
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * dim * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < dim; b++)
     syn1neg[a * dim + b] = 0;// init parameter vector syn1neg
  }

  for (a = 0; a < vocab_size; a++) for (b = 0; b < dim; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * dim + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dim;// init word vector syn0
  }

  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(dim, sizeof(real)); // x_w
  real *neu1e = (real *)calloc(dim, sizeof(real)); // e
  //modification begin
  real *morpheme = (real *)calloc(dim, sizeof(real));
  real *prefixComp = (real *)calloc(dim, sizeof(real));
  real *rootComp = (real *)calloc(dim, sizeof(real));
  real *suffixComp = (real *)calloc(dim, sizeof(real));

  int pCnt, rCnt, sCnt; // count of each morpheme
  int curIdx;
  //modification end
  FILE *fi = fopen(train_file, "rb"); // binary file type
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET); // int fseek(FILE *stream, long offset, int fromwhere); fromwhere->SEEK_CUR|SEEK_END|SEEK_SET:current|end|begin
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1)); // update learning rate
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001; // guarantee the minimum learning rate
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < dim; c++) neu1[c] = 0;
    for (c = 0; c < dim; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;

    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++){ 
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;

          //modification begin
          for (c = 0; c < dim; c++) morpheme[c] = 0;
          for (c = 0; c < dim; c++) morpheme[c] = syn0[c + last_word * dim];

          for (c = 0; c < dim; c++) { prefixComp[c] = 0; rootComp[c] = 0; suffixComp[c] = 0; }

          pCnt = vocab[last_word].pn; 
          rCnt = vocab[last_word].rn;
          sCnt = vocab[last_word].sn;

          if(pCnt != 0){
            for(curIdx = 0; curIdx < pCnt; curIdx++){
              long long prefixWord = vocab[last_word].prefix[curIdx].position;
              //printf("[Debug] TrainModelThread-prefix: %lld\n", prefixWord);
              for (c = 0; c < dim; c++) prefixComp[c] +=  syn0[c + prefixWord * dim];
            }
          }

          if(rCnt != 0){
            for(curIdx = 0; curIdx < rCnt; curIdx++){
              long long rootWord = vocab[last_word].root[curIdx].position;
              //printf("[Debug] TrainModelThread-root: %lld\n", rootWord);
              for (c = 0; c < dim; c++) rootComp[c] +=  syn0[c + rootWord * dim];
            }
          }

          if(sCnt != 0){
            for(curIdx = 0; curIdx < sCnt; curIdx++){
              long long suffixWord = vocab[last_word].suffix[curIdx].position;
              //printf("[Debug] TrainModelThread-suffix: %lld\n", suffixWord);
              for (c = 0; c < dim; c++) suffixComp[c] +=  syn0[c + suffixWord * dim];
            }
          }
		  
          int norm = 1;
          if(pCnt + rCnt + sCnt != 0){
            for (c = 0; c < dim; c++)
              morpheme[c] += (prefixComp[c] + rootComp[c] + suffixComp[c]) / (pCnt + rCnt + sCnt); //wegihted averaging
            norm = 2;
          }

          for (c = 0; c < dim; c++) neu1[c] += morpheme[c] / norm;
	        //modification end
          cw++;
        }
      }
      if (cw) { // CBOW
        for (c = 0; c < dim; c++) neu1[c] /= cw;
	      // HIERACHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * dim;
          // Propagate hidden -> output
          for (c = 0; c < dim; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < dim; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < dim; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * dim;
          f = 0;
          for (c = 0; c < dim; c++) f += neu1[c] * syn1neg[c + l2]; //x^T_w * theta^u

          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

          for (c = 0; c < dim; c++) neu1e[c] += g * syn1neg[c + l2]; //e := e + g * theta^u
          for (c = 0; c < dim; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < dim; c++) syn0[c + last_word * dim] += neu1e[c];

          //modification begin
          pCnt = vocab[last_word].pn; 
          rCnt = vocab[last_word].rn;
          sCnt = vocab[last_word].sn;

          if(pCnt != 0){
            for(curIdx = 0; curIdx < pCnt; curIdx++){
              long long prefixWord = vocab[last_word].prefix[curIdx].position;
              for (c = 0; c < dim; c++) syn0[c + prefixWord * dim] += neu1e[c];
            }
          }

          if(rCnt != 0){
            for(curIdx = 0; curIdx < rCnt; curIdx++){
              long long rootWord = vocab[last_word].root[curIdx].position;
              for (c = 0; c < dim; c++) syn0[c + rootWord * dim] += neu1e[c];
            }
          }

          if(sCnt != 0){
            for(curIdx = 0; curIdx < sCnt; curIdx++){
              long long suffixWord = vocab[last_word].suffix[curIdx].position;
              for (c = 0; c < dim; c++) syn0[c + suffixWord * dim] += neu1e[c];
            }
          }
          //modification end
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * dim;

        for (c = 0; c < dim; c++) neu1e[c] = 0;

        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * dim;
          // Propagate hidden -> output
          for (c = 0; c < dim; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < dim; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < dim; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * dim;
          f = 0;
          for (c = 0; c < dim; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < dim; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < dim; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < dim; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  //modification begin
  if (wordmap_file[0] == 0 || output_file[0] == 0) return;
  //modification end
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a); //create num_threads training thread
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
	  //modification begin
    //fprintf(fo, "%lld %lld\n", vocab_size, dim); //disable the input of vocab_size and dim
	  //modification end
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < dim; b++) fwrite(&syn0[a * dim + b], sizeof(real), 1, fo);
      else for (b = 0; b < dim; b++) fprintf(fo, "%lf ", syn0[a * dim + b]);
      fprintf(fo, "\n");
    }

  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * dim, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * dim; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < dim; d++) cent[dim * cl[c] + d] += syn0[c * dim + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < dim; c++) {
          cent[dim * b + c] /= centcn[b];
          closev += cent[dim * b + c] * cent[dim * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < dim; c++) cent[dim * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < dim; b++) x += cent[dim * d + b] * syn0[c * dim + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  //modification begin
  //FreeMap();
  //modification end
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    //modification begin
    printf("\t-refvocab <file>\n");
    printf("\t\tUse reference vocabulary from <file> to calculate cosine similarity\n");
    printf("\t-wordmap <file>\n");
    printf("\t\tUse text data from <file> to map the target word\n");
    //modification end
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    //modification begin
    printf("./word2vec -train data.txt -wordmap wordmap.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    //modification end
    return 0;
  }
  //modification begin
  wordmap_file[0] = 0;
  //modification end
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  //modification begin
  if ((i = ArgPos((char *)"-wordmap", argc, argv)) > 0) strcpy(wordmap_file, argv[i + 1]);
  //modification end
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = 1/(1+e^(-x)) = e^x / (e^x + 1)
  }
  TrainModel();
  return 0;
}
