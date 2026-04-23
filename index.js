const express = require("express");
const Sentiment = require("sentiment");
const nlp = require("compromise");
const natural = require("natural");

const app = express();
app.use(express.json());
app.use(express.static("public"));

const sentiment = new Sentiment();
const tokenizer = new natural.WordTokenizer();
const TfIdf = natural.TfIdf;

function analyze(text) {
  const doc = nlp(text);

  //Tokenization – split text into words
  const tokens = tokenizer.tokenize(text);

  const sentimentResult = sentiment.analyze(text);
  const sentimentLabel =
    sentimentResult.score > 0
      ? "Positive"
      : sentimentResult.score < 0
      ? "Negative"
      : "Neutral";

  const people = doc.people().out("array");
  const places = doc.places().out("array");
  const organizations = doc.organizations().out("array");

  const nouns = doc.nouns().out("array");
  const verbs = doc.verbs().out("array");
  const adjectives = doc.adjectives().out("array");

  const tfidf = new TfIdf();
  tfidf.addDocument(text);
  const keywords = [];
  tfidf.listTerms(0).slice(0, 5).forEach((item) => keywords.push(item.term));

  //Word frequency
  const freq = {};
  tokens.forEach((w) => {
    const word = w.toLowerCase();
    freq[word] = (freq[word] || 0) + 1;
  });
  const topWords = Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([word, count]) => ({ word, count }));

  //Stemming – reduce words to root form
  const stems = tokens.slice(0, 8).map((w) => ({
    word: w,
    stem: natural.PorterStemmer.stem(w),
  }));

  return {
    input: text,
    stats: {
      characters: text.length,
      words: tokens.length,
      sentences: text.split(/[.!?]+/).filter(Boolean).length,
    },
    sentiment: {
      label: sentimentLabel,
      score: sentimentResult.score,
      positive: sentimentResult.positive,
      negative: sentimentResult.negative,
    },
    namedEntities: { people, places, organizations },
    partsOfSpeech: { nouns, verbs, adjectives },
    keywords,
    topWords,
    stemming: stems,
  };
}

app.get("/", (req, res) => res.sendFile(__dirname + "/public/index.html"));

app.post("/analyze", (req, res) => {
  const { text } = req.body;
  if (!text || text.trim() === "")
    return res.status(400).json({ error: "Please provide text to analyze" });
  res.json(analyze(text));
});

app.listen(3000, () => {
  console.log("🚀 NLP Text Analyzer running at http://localhost:3000");
});
