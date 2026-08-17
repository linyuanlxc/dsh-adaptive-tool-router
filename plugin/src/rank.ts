export interface ToolSchema {
  name: string
  description?: string
  parameters?: unknown
}

export interface RankedTool {
  name: string
  score: number
  rank: number
}

const TOKEN_PATTERN = /[\p{L}\p{N}_]+/gu

export function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_PATTERN) ?? []
}

export function retrievalText(schema: ToolSchema): string {
  const parts = [schema.name, schema.description ?? '']
  if (schema.parameters !== undefined) {
    parts.push(JSON.stringify(schema.parameters))
  }
  return parts.filter(Boolean).join('\n')
}

export function rankTools(
  query: string,
  schemas: readonly ToolSchema[],
  options: { limit: number; k1?: number; b?: number } = { limit: 10 },
): RankedTool[] {
  const { limit, k1 = 1.5, b = 0.75 } = options
  if (limit <= 0) {
    throw new Error('limit must be positive')
  }
  if (schemas.length === 0) {
    return []
  }

  const documents = schemas.map(schema => tokenize(retrievalText(schema)))
  const termCounts = documents.map(countTerms)
  const lengths = documents.map(document => document.length)
  const avgLength = lengths.reduce((sum, length) => sum + length, 0) / lengths.length
  const documentFrequency = new Map<string, number>()
  for (const document of documents) {
    for (const term of new Set(document)) {
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1)
    }
  }

  const size = schemas.length
  const idf = new Map<string, number>()
  for (const [term, frequency] of documentFrequency) {
    idf.set(term, Math.log(1 + (size - frequency + 0.5) / (frequency + 0.5)))
  }

  const queryTerms = tokenize(query)
  const scored = schemas.map((schema, index) => {
    const counts = termCounts[index]
    const documentLength = lengths[index]
    const normalization = 1 - b + b * documentLength / avgLength
    let score = 0
    for (const term of queryTerms) {
      const frequency = counts.get(term) ?? 0
      if (!frequency) continue
      const numerator = frequency * (k1 + 1)
      const denominator = frequency + k1 * normalization
      score += (idf.get(term) ?? 0) * numerator / denominator
    }
    return { name: schema.name, score }
  })

  scored.sort((left, right) => right.score - left.score || left.name.localeCompare(right.name))
  return scored.slice(0, limit).map((item, index) => ({
    ...item,
    rank: index + 1,
  }))
}

function countTerms(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1)
  }
  return counts
}
