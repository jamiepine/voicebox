/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ProfileSampleResponse = {
  description: `Response model for profile sample.`,
  properties: {
    id: {
      type: 'string',
      isRequired: true,
    },
    profile_id: {
      type: 'string',
      isRequired: true,
    },
    audio_path: {
      type: 'string',
      isRequired: true,
    },
    reference_text: {
      type: 'string',
      isRequired: true,
    },
    sort_order: {
      type: 'number',
      isRequired: true,
    },
  },
} as const;
