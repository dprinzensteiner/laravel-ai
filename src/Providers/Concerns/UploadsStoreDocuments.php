<?php

namespace Laravel\Ai\Providers\Concerns;

use Laravel\Ai\Contracts\Files\StorableFile;
use Laravel\Ai\Contracts\Gateway\StoreGateway;
use Laravel\Ai\Contracts\Gateway\UploadsDocuments;

trait UploadsStoreDocuments
{
    /**
     * Upload a document's contents directly into a vector store.
     */
    public function uploadDocumentToStore(string $storeId, StorableFile $file, array $metadata = []): string
    {
        /** @var StoreGateway&UploadsDocuments $gateway */
        $gateway = $this->storeGateway();

        return $this->dispatchFileAddition(
            $storeId, null,
            fn () => $gateway->uploadDocument($this, $storeId, $file, $metadata),
        );
    }
}
