import { Callout } from 'fumadocs-ui/components/callout';
import { Card, Cards } from 'fumadocs-ui/components/card';
import { File, Files, Folder } from 'fumadocs-ui/components/files';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import type { ReactNode } from 'react';
import { APIPage } from '@/components/api-page';

// Simple accordion using native HTML details/summary
function AccordionGroup({ children }: { children: ReactNode }) {
  return <div className="my-6 space-y-2">{children}</div>;
}

function Accordion({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="group border rounded-lg p-4">
      <summary className="cursor-pointer font-semibold list-none">
        <span className="group-open:rotate-90 transition-transform inline-block mr-2">▶</span>
        {title}
      </summary>
      <div className="mt-4 pl-6">{children}</div>
    </details>
  );
}

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    // Layout components
    Card,
    Cards,
    // Files
    Files,
    Folder,
    File,
    // Callouts
    Callout,
    // Tabs
    Tabs,
    Tab,
    // Steps
    Steps,
    Step,
    // Accordion (native HTML-based)
    AccordionGroup,
    Accordion,
    // OpenAPI component
    APIPage,
    ...components,
  };
}
