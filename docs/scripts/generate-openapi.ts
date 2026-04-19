import { generateFiles } from 'fumadocs-openapi';
import { openapi } from '../lib/openapi';

await generateFiles({
  input: openapi,
  output: 'content/docs/api-reference',
  groupBy: 'tag',
});

console.log('✓ OpenAPI documentation generated in content/docs/api-reference/');
