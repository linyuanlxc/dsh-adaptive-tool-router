export function queryFromMessages(messages: readonly unknown[]): string {
  return messages
    .map(message => extractText((message as { content?: unknown }).content))
    .filter(Boolean)
    .join('\n')
}

function extractText(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    return content
      .map(part => {
        if (typeof part === 'string') return part
        if (part && typeof part === 'object' && 'text' in part) {
          return String((part as { text?: unknown }).text ?? '')
        }
        return ''
      })
      .filter(Boolean)
      .join('\n')
  }
  if (content == null) {
    return ''
  }
  return typeof content === 'object' ? JSON.stringify(content) : String(content)
}
