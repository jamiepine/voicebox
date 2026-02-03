import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { APIPage } from '@/components/api-page';
import {
  Accordion,
  AccordionGroup,
  CardGroup,
  Danger,
  Frame,
  Info,
  MintlifyCard,
  Note,
  Step,
  Steps,
  Tip,
  Warning,
} from '@/components/mintlify-compat';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    // Mintlify compatibility components
    Frame,
    CardGroup,
    Card: MintlifyCard,
    Steps,
    Step,
    Tip,
    Note,
    Warning,
    Info,
    Danger,
    AccordionGroup,
    Accordion,
    // OpenAPI component
    APIPage,
    ...components,
  };
}
