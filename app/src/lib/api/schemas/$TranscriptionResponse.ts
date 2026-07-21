/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $TranscriptionResponse = {
  description: `Response model for transcription.`,
  properties: {
    text: {
      type: 'string',
      isRequired: true,
    },
    duration: {
      type: 'number',
      isRequired: true,
    },
    language: {
      type: 'any-of',
      contains: [{ type: 'string' }, { type: 'null' }],
    },
  },
} as const;
