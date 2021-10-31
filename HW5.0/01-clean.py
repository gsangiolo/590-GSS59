# I don't have a browser window in the linux box I'm working on, so I'm using the Gutenberg examples from NLTK
import wikipedia
import nltk
#nltk.download('gutenberg')
from nltk.corpus import gutenberg
import re, math, json, random

# The Cleaner Class can be used to process text data from one of 3 sources: NLTK's Gutenberg Dataset, the local filesystem, or Wikipedia.
# Currently, the Wikipedia function only supports querying for "QUERY X", where QUERY can be provided, and X is a list of random-ish words provided in the gather_text function below. Future iterations can allow for more complex querying, but I wanted to keep this simple for now
class Cleaner:

    # Initialize the Cleaner. Here you can specify 3 things:
    # max_texts: The max documents to be gathered for gutenberg or text_file modes. Does not apply to Wiki
    # n_splits: The number of splits to divide each text into. This is for processing texts and trying to identify which document it belongs to
    # mode: Has 3 possibilities: gutenberg, text_file, or wiki. Determines the source of text data
    def __init__(self, max_texts=20, n_splits=10, mode='gutenberg'):
        self.max_texts = max_texts
        self.n_splits = n_splits
        self.texts = {}
        self.mode = mode
        if self.mode == 'gutenberg':
            nltk.download('gutenberg')

    # Worker function that gathers texts from the source that this has been initialized with. Can provide 1 argument: wiki_query: for wiki mode, will append this term to the list of prefixes specified (the misnamed countries_list variable)
    def gather_text(self, wiki_query='food'):
        # If I weren't using this pre-split text, I would use something like:
        if self.mode == 'text_file':
            count = 0
            filenames = os.listdir('./data')
            random.shuffle(filenames)
            # Read files in the directory until you reach max_texts
            for name in filenames:
                if count >= self.max_texts:
                    break
                with open(os.path.join('./data', name)) as f:
                    text = [word for line in f for word in line.split()]
                    self.texts[name] = text
                    count += 1
        

        elif self.mode == 'gutenberg':
            filenames = gutenberg.fileids()
                        
            filenames = filenames[:self.max_texts*10]

            random.shuffle(filenames)
            # From a set of Gutenberg texts, read only max_texts of them (randomly shuffled)
            for name in filenames:
                text = gutenberg.words(name)
                # Regex expression to remove non-alpha words, from: https://stackoverflow.com/questions/46486157/how-to-remove-every-word-with-non-alphabetic-characters
                self.texts[name] = [word.lower() for word in text if re.match(r'[^\W\d]*$', word)]
        elif self.mode == 'wiki':
            #------------------------
            #QUERY WIKI -- adapted from Lecture codes!
            #------------------------
            # Really a query suffix list, but that's ok!
            country_list = ['mexican']#, 'spanish', 'japanese','references','china','chinese','external', 'somalian','citation', 'korean', 'chad','brazilian',]
            stop_words=['']

            for country in country_list:

                # topic='food in '+country
                topic = wiki_query + ' ' + country
            
                #--------------------------
                #SEARCH FOR RELEVANT PAGES
                #--------------------------
                titles=wikipedia.search(topic,results=self.max_texts)
                print("TITLES=",titles)
    
                #FUNCTION TO PRINT BASIC ABOUT WIKI PAGE
#                def print_info(wiki_page):
#                    print("-------------------------")
#                    print(wiki_page.title)
#                    print(wiki_page.url)
                    # print(wiki_page.sections)
        
#                    if(verbose):
#                        print(wiki_page.sections)
#                        print(wiki_page.categories)
#                        print(wiki_page.html)
#                        print(wiki_page.images)
#                        print(wiki_page.content)
#                        print(wikipedia.summary(wiki_page.title, auto_suggest=False))
#                        print(wiki_page.references)
#                        print(wiki_page.links[0],len(page.links))

                #--------------------------
                #LOOP OVER TITLES
                #--------------------------
                num_files=0
                sections=[]

                for title in titles:
                    try:
                        page = wikipedia.page(title, auto_suggest=False)
                        #print_info(page)

                        sections=sections+page.sections
                        num_files+=1
                    except:
                        print("SOMETHING WENT WRONG:", title);

                #CONVERT TO ONE LONG STRING
                text=''
                for string in sections:
                    words=string.lower().split()
                    for word in words:
                        if(word not in stop_words):
                                text=text+word+' '
                if len(text) > 0: # Some queries came back empty, which may be some rate-limits that I ran into? Either way it messes up the pipeline later
                    self.texts[topic] = text.split()
        return self.texts

    # Splits text data into self.n_splits chunks -- for classifying text chunks into their corresponding documents.
    # target_dir: specifies where the data should be written. self.mode will be appended to this
    # write: specifies if the data should be written to the filesystem at all!Data will be written as JSON files
    def split_chunks(self, target_dir='./data_clean/', write=True):
        all_chunks = {}
        # Split each text into chunks!
        for name, text in self.texts.items():
            len_text = len(text)
            chunk_size = math.floor(len_text / self.n_splits)
            chunks = []
            for i in range(self.n_splits):
                chunks.append(text[chunk_size*i:(i+1)*chunk_size if (i+1)*chunk_size < len_text else len_text])
            if write:
                with open(target_dir + self.mode + '/' + name + '.json', 'w') as f:
                    f.write(json.dumps(chunks))
            all_chunks[name] = chunks
        return all_chunks
