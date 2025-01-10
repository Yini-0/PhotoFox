import Foundation
import NaturalLanguage

enum EmbeddingUtils {
    static func generateEmbedding(for text: String) -> [Double] {
        guard let embedder = NLEmbedding.wordEmbedding(for: .english) else {
            return []
        }
        
        // Tokenize the text
        let tagger = NLTagger(tagSchemes: [.tokenType])
        tagger.string = text.lowercased()
        
        // Get embeddings for each word and average them
        var embeddings: [[Double]] = []
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .tokenType) { _, range in
            let word = String(text[range]).lowercased()
            if let vector = embedder.vector(for: word) {
                embeddings.append(vector)
            }
            return true
        }
        
        // Average the embeddings
        guard !embeddings.isEmpty else { return [] }
        let dimension = embeddings[0].count
        var averageEmbedding = Array(repeating: 0.0, count: dimension)
        
        for embedding in embeddings {
            for (i, value) in embedding.enumerated() {
                averageEmbedding[i] += value
            }
        }
        
        return averageEmbedding.map { $0 / Double(embeddings.count) }
    }
    
    static func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        
        let dotProduct = zip(a, b).map(*).reduce(0, +)
        let magnitudeA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let magnitudeB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        
        guard magnitudeA > 0, magnitudeB > 0 else { return 0 }
        return dotProduct / (magnitudeA * magnitudeB)
    }
}
