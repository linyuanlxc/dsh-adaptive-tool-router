import { describe, expect, it } from 'vitest'
import { queryFromMessages } from '../src/query.ts'
import { rankTools } from '../src/rank.ts'

describe('rankTools', () => {
  it('ranks the relevant tool first', () => {
    const ranking = rankTools(
      'Will it rain in Beijing? weather forecast',
      [
        {
          name: 'weather_forecast',
          description: 'Get a weather forecast for a city',
          parameters: { city: { type: 'string' } },
        },
        {
          name: 'stock_price',
          description: 'Get the current stock market price',
          parameters: { ticker: { type: 'string' } },
        },
      ],
      { limit: 2 },
    )

    expect(ranking[0]?.name).toBe('weather_forecast')
    expect(ranking[0]?.score).toBeGreaterThan(ranking[1]?.score ?? 0)
  })

  it('returns an empty list when the catalog is empty', () => {
    expect(rankTools('weather', [], { limit: 3 })).toEqual([])
  })
})

describe('queryFromMessages', () => {
  it('joins user text parts', () => {
    expect(queryFromMessages([
      { content: 'Look up Beijing weather' },
      { content: [{ type: 'text', text: 'tomorrow morning' }] },
    ])).toBe('Look up Beijing weather\ntomorrow morning')
  })
})
